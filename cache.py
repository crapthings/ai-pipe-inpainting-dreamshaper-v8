import torch
from diffusers import StableDiffusionInpaintPipeline
from compel import Compel

from config import model_name

print('cache model')

pipe = StableDiffusionInpaintPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

compel = Compel(tokenizer = pipe.tokenizer, text_encoder = pipe.text_encoder)

print('done')
