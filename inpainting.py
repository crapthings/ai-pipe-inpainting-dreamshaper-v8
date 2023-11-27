import torch
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForImage2Image
from diffusers import EulerAncestralDiscreteScheduler

from config import model_name

def sc(self, clip_input, images): return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc

inpaintingPipe = StableDiffusionInpaintPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

inpaintingPipe.scheduler = EulerAncestralDiscreteScheduler.from_config(inpaintingPipe.scheduler.config)

inpaintingPipe.to('cuda')

img2imgPipe = AutoPipelineForImage2Image.from_pipe(txt2imgPipe)

def inpainting (**props):
  output = inpaintingPipe(**props)
  return output

def img2img (**props):
  output = img2imgPipe(**props)
  return output
