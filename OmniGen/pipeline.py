import os
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Literal
import gc

from PIL import Image
import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from diffusers.models import AutoencoderKL
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from transformers import BitsAndBytesConfig
from safetensors.torch import load_file

from OmniGen import OmniGen, OmniGenProcessor, OmniGenScheduler


logger = logging.get_logger(__name__) 

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from OmniGen import OmniGenPipeline
        >>> pipe = FluxControlNetPipeline.from_pretrained(
        ...     base_model
        ... )
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""

def best_available_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        logger.info("Don't detect any available GPUs, using CPU instead, this may take long time to generate image!!!")
        device = torch.device("cpu")
    return device

def print_mem(msg:str = None):
    free,tot = torch.cuda.mem_get_info()
    max_reser = torch.cuda.max_memory_reserved()
    max_alloc = torch.cuda.max_memory_allocated()
    
    def fmt_bytes(nbytes, mb=True):
        prec = 2 if mb else 3
        return f'{nbytes/(1024**prec):0.2f} ({nbytes/(1000**prec):0.2f})'
    
    if msg:
        print(f'## {msg} ##')
    print(f'Cur Used: {fmt_bytes(tot-free)}, max_memory_reserved: {fmt_bytes(max_reser)}, max_memory_allocated: {fmt_bytes(max_alloc)}')

class OmniGenPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGen,
        processor: OmniGenProcessor,
        device: Union[str, torch.device] = None,
    ):
        self.vae = vae
        self.model = model
        self.processor = processor
        self.device = device

        if self.device is None:
            self.device = best_available_device()
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)

        # self.model.to(torch.bfloat16)
        self.model.eval()
        self.vae.eval()

        self.model_cpu_offload = False

    @classmethod
    def from_pretrained(cls, model_name, vae_path: str=None, device=None, low_cpu_mem_usage=True, **kwargs):
        pretrained_path = Path(model_name)
        
        # XXX: Consider renaming 'model' to 'transformer' conform to diffusers pipeline syntax
        model = kwargs.get('model', None)
        processor = kwargs.get('processor', None)
        vae = kwargs.get('vae', None)

        # NOTE: should technically allow delayed component inits via model/vae = None, but seems like more of a footgun than it's worth at this point
        
        if not pretrained_path.exists():
            ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'model.pt']

            if model is not None:
                ignore_patterns.append('model.safetensors') # avoid downloading bf16 model if passing existing model
            
            logger.info("Model not found, downloading...")
            cache_folder = os.getenv('HF_HUB_CACHE')
            pretrained_path = Path(snapshot_download(repo_id=model_name, cache_dir=cache_folder, ignore_patterns=ignore_patterns))
            logger.info(f"Downloaded model to {pretrained_path}")
        
        if model is None:
            model = OmniGen.from_pretrained(pretrained_path, dtype=torch.bfloat16, quantization_config=None, low_cpu_mem_usage=low_cpu_mem_usage)
        
        model = model.requires_grad_(False).eval()

        if processor is None:
            processor = OmniGenProcessor.from_pretrained(model_name)

        if vae is None:
            if vae_path is None:
                vae_path = pretrained_path.joinpath("vae")
            
            if not os.path.exists(vae_path):
                logger.info(f"No VAE found in {model_name}, downloading stabilityai/sdxl-vae from HF")
                vae_path = "stabilityai/sdxl-vae"
                
            vae = AutoencoderKL.from_pretrained(vae_path, low_cpu_mem_usage=low_cpu_mem_usage)

        return cls(vae, model, processor, device)
    
    def merge_lora(self, lora_path: str):
        model = PeftModel.from_pretrained(self.model, lora_path)
        model.merge_and_unload()

        self.model = model
    
    def to(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self.vae.to(device)
        self.device = device

    def vae_encode(self, x, dtype):
        x = self.vae.encode(x).latent_dist.sample()
        if self.vae.config.shift_factor is not None:
            x -= self.vae.config.shift_factor
        
        return (x*self.vae.config.scaling_factor).to(dtype)
        # if self.vae.config.shift_factor is not None:
        #     x = self.vae.encode(x).latent_dist.sample()
        #     x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        # else:
        #     x = self.vae.encode(x).latent_dist.sample().mul_(self.vae.config.scaling_factor)
        # x = x.to(dtype)
        # return x
    
    def move_to_device(self, data):
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)

    def enable_model_cpu_offload(self):
        self.model_cpu_offload = True
        if not self.model.quantized:
            self.model.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()  # Clear VRAM
        gc.collect()  # Run garbage collection to free system RAM
    
    def disable_model_cpu_offload(self):
        self.model_cpu_offload = False
        self.model.to(self.device)
        self.vae.to(self.device)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = None,
        output_type: str = "pil",
        ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation. 
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the 1-th image in list.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. The number must be a multiple of 16.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. The number must be a multiple of 16.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            max_input_image_size (`int`, *optional*, defaults to 1024): the maximum size of input image, which will be used to crop the input image to the maximum size
            separate_cfg_infer (`bool`, *optional*, defaults to False):
                Perform inference on images with different guidance separately; this can save memory when generating images of large size at the expense of slower inference.
            use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            offload_kv_cache (`bool`, *optional*, defaults to True): offload the cached key and value to cpu, which can save memory but slow down the generation silightly
            offload_model (`bool`, *optional*, defaults to False): offload the model to cpu, which can save memory but slow down the generation
            use_input_image_size_as_output (bool, defaults to False): whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task
            seed (`int`, *optional*):
                A random seed for generating output. 
            dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                data type for the model
            output_type (`str`, *optional*, defaults to "pil"):
                The type of the output image, which can be "pt" or "pil"
        Examples:

        Returns:
            A list with the generated images.
        """
        # check inputs:
        if use_input_image_size_as_output:
            assert isinstance(prompt, str) and len(input_images) == 1, "if you want to make sure the output image have the same size as the input image, please only input one image instead of multiple input images"
        else:
            assert height%16 == 0 and width%16 == 0, "The height and width must be a multiple of 16."
        if input_images is None:
            use_img_guidance = False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None
        

        # set model and processor
        if max_input_image_size != self.processor.max_image_size:
            self.processor = OmniGenProcessor(self.processor.text_tokenizer, max_image_size=max_input_image_size)
        
        if not self.model.quantized:
            self.model.dtype = dtype
            self.model.to(dtype)
        if self.model_cpu_offload and separate_cfg_infer:
            self.vae = self.vae.to(dtype) # Uncomment this line to allow bfloat16 VAE
        # self.vae.enable_tiling()
        # self.vae.enable_slicing()
        if offload_model:
            self.enable_model_cpu_offload()
        else:
            self.disable_model_cpu_offload()

        input_data = self.processor(prompt, input_images, height=height, width=width, use_img_cfg=use_img_guidance, separate_cfg_input=separate_cfg_infer, use_input_image_size_as_output=use_input_image_size_as_output)

        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1
        if use_input_image_size_as_output:
            if separate_cfg_infer:
                height, width = input_data['input_pixel_values'][0][0].shape[-2:]
            else:
                height, width = input_data['input_pixel_values'][0].shape[-2:]
        
        latent_size_h, latent_size_w = height//8, width//8

        # latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, dtype=self.model.dtype, device=self.device, generator=generator)
        # latents = torch.cat([latents]*(1+num_cfg), 0).to(self.model.dtype)
        print_mem('Before Vae Encode')
        if input_images is not None and self.model_cpu_offload: 
            self.vae.to(self.device)
        
        input_img_latents = []
        if separate_cfg_infer:
            _device = 'cpu' if self.model_cpu_offload else self.device
            for temp_pixel_values in input_data['input_pixel_values']:
                input_img_latents.append([self.vae_encode(img.to(self.vae.device, self.vae.dtype), self.model.dtype).to(_device) for img in temp_pixel_values])
        else:
            for img in input_data['input_pixel_values']:
                input_img_latents.append(self.vae_encode(img.to(self.vae.device, self.vae.dtype), self.model.dtype))
                
        
        if input_images is not None and self.model_cpu_offload:
            self.vae.to('cpu')
            torch.cuda.empty_cache()  # Clear VRAM
            gc.collect()  # Run garbage collection to free system RAM

        print_mem('After Vae Encode')

        model_kwargs = dict(
            # input_ids=self.move_to_device(input_data['input_ids']), 
            input_ids=input_data['input_ids'], 
            input_img_latents=input_img_latents, 
            input_image_sizes=input_data['input_image_sizes'], 
            # attention_mask=self.move_to_device(input_data["attention_mask"]), 
            # position_ids=self.move_to_device(input_data["position_ids"]), 
            attention_mask=input_data["attention_mask"], 
            position_ids=input_data["position_ids"], 
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
            use_kv_cache=use_kv_cache,
            offload_model=offload_model,
        )
        print_mem('After model_kwargs')
        if separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg

        if self.model_cpu_offload and not self.model.quantized:
            for name, param in self.model.named_parameters():
                device = 'cpu' if 'layers' in name and 'layers.0' not in name else self.device
                param.data = param.data.to(device)

            for buffer_name, buffer in self.model.named_buffers():
                setattr(self.model, buffer_name, buffer.to(self.device))
            torch.cuda.empty_cache()  
            gc.collect()  
        # else:
        #     self.model.to(self.device)
        print_mem('After param offload')
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, dtype=self.model.dtype, device=self.device, generator=generator).repeat(1+num_cfg, 1, 1, 1)
        #latents = torch.cat([latents]*(1+num_cfg), 0).to(self.model.dtype)
        print_mem('After latents')
        scheduler = OmniGenScheduler(num_steps=num_inference_steps)
        samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache, offload_kv_cache=offload_kv_cache).chunk((1+num_cfg), dim=0)[0]
        print_mem('After scheduler')
        if self.model_cpu_offload:
            for name, param in self.model.named_parameters():
                param.data = param.data.to('cpu')
            #if not self.model.quantized:
            #    self.model.to("cpu")

            torch.cuda.empty_cache()  
            gc.collect()  

        print_mem('Before samples to float32')
        self.vae.to(self.device)
        samples = samples.to(self.vae.dtype) / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            samples += self.vae.config.shift_factor
        # if self.vae.config.shift_factor is not None:
        #     samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
        # else:
        #     samples = samples / self.vae.config.scaling_factor   
        if self.model_cpu_offload:
            del latents, model_kwargs, scheduler, input_img_latents, input_data, input_images
            torch.cuda.empty_cache()  
            gc.collect()  
        
        print_mem('Before vae decode')
        samples = self.vae.decode(samples).sample
        print_mem('After vae decode')

        if self.model_cpu_offload:
            self.vae.to('cpu')
            torch.cuda.empty_cache()  
            gc.collect()  
        
        samples = (samples * 0.5 + 0.5).clamp(0, 1)

        if output_type == "pt":
            output_images = samples
        else:
            output_samples = (samples * 255).to("cpu", dtype=torch.uint8).permute(0, 2, 3, 1).numpy()
            # output_samples = output_samples.permute(0, 2, 3, 1).numpy()
            output_images = [Image.fromarray(sample) for sample in output_samples]
            # output_images = []
            # for i, sample in enumerate(output_samples):
            #     output_images.append(Image.fromarray(sample))

        torch.cuda.empty_cache()  # Clear VRAM
        gc.collect()              # Run garbage collection to free system RAM

        return output_images
