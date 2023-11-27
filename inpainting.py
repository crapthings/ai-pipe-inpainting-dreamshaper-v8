import torch
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from diffusers import EulerAncestralDiscreteScheduler

from config import inpainting_model_name, model_name

def sc(self, clip_input, images): return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc

inpaintingPipe = StableDiffusionInpaintPipeline.from_single_file(
  inpainting_model_name,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

inpaintingPipe.scheduler = EulerAncestralDiscreteScheduler.from_config(inpaintingPipe.scheduler.config)

# inpaintingPipe.to('cuda')
inpaintingPipe.enable_model_cpu_offload()

# img2imgPipe = StableDiffusionImg2ImgPipeline.from_single_file(
#     model_name,
#     torch_dtype = torch.float16,
#     variant = 'fp16',
#     use_safetensors = True
# )

# img2imgPipe.to('cuda')

# img2imgPipe.scheduler = EulerAncestralDiscreteScheduler.from_config(img2imgPipe.scheduler.config)

def inpainting (**props):
  output = inpaintingPipe(**props)
  return output

# def img2img (**props):
#   output = img2imgPipe(**props)
#   return output
