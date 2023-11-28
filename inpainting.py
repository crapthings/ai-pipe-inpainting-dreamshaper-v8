import torch
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers import StableDiffusionInpaintPipeline
from diffusers import EulerAncestralDiscreteScheduler
from compel import Compel

from config import model_name, model_name
from utils import sc

safety_checker.StableDiffusionSafetyChecker.forward = sc

inpaintingPipe = StableDiffusionInpaintPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

inpaintingPipe.scheduler = EulerAncestralDiscreteScheduler.from_config(inpaintingPipe.scheduler.config)

inpaintingPipe.to('cuda')
# inpaintingPipe.enable_model_cpu_offload()

compel = Compel(tokenizer = inpaintingPipe.tokenizer, text_encoder = inpaintingPipe.text_encoder)

def inpainting (**props):
  output = inpaintingPipe(**props)
  return output
