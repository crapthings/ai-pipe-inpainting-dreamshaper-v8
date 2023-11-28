import torch
from diffusers import StableDiffusionPipeline

from config import model_name

print('cache model')

StableDiffusionPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

print('done')
