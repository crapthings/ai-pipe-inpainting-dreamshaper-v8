import math
import requests

import numpy as np
import torch
from diffusers.utils import load_image
from PIL import Image, ImageOps
import runpod

from utils import extract_origin_pathname, upload_image, rounded_size, zoom_and_crop, open_image
# from inpainting import inpainting, img2img
from inpainting import inpainting

def run (job, _generator = None):
    # prepare task
    try:
        print('debug', job)

        _input = job.get('input')

        debug = _input.get('debug')
        upload_url = _input.get('upload_url')
        input_url = _input.get('input_url')
        mask_url = _input.get('mask_url')

        prompt = _input.get('prompt', 'a dog')
        negative_prompt = _input.get('negative_prompt', '')
        # width = int(np.clip(_input.get('width', 768), 256, 1024))
        # height = int(np.clip(_input.get('height', 768), 256, 1024))
        num_inference_steps = int(np.clip(_input.get('num_inference_steps', 30), 20, 150))
        guidance_scale = float(np.clip(_input.get('guidance_scale', 13.0), 0, 30))
        seed = _input.get('seed')
        upscale = _input.get('upscale')
        strength = _input.get('strength')

        if strength is not None:
            strength = float(np.clip(strength, 0, 1))

        if upscale is not None:
            upscale = float(np.clip(upscale, 1, 4))

        # input_image = load_image(input_url).convert('RGBA')
        input_image = open_image(input_url)
        mask_image = load_image(mask_url)

        limit = 960
        input_image.thumbnail([limit, limit])
        mask_image.thumbnail([limit, limit])

        mask_image = ImageOps.invert(mask_image)

        mask_image = zoom_and_crop(mask_image, 1.1)

        renderWidth, renderHeight = rounded_size(input_image.width, input_image.height)

        if seed is not None:
            _generator = torch.Generator(device = 'cuda').manual_seed(seed)

        output_image = inpainting(
            image = input_image.convert('RGB'),
            mask_image = mask_image,
            prompt = prompt,
            negative_prompt = negative_prompt,
            width = renderWidth,
            height = renderHeight,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            strength = strength,
            generator = _generator
        ).images[0]

        output_image = output_image.resize(input_image.size).convert('RGBA')

        # if upscale is not None:
        #     output_image = output_image.resize([int(input_image.width * upscale), int(input_image.height * upscale)])

        # if strength is not None:
        #     output_image = img2img(
        #         image = output_image,
        #         prompt = prompt,
        #         negative_prompt = negative_prompt,
        #         num_inference_steps = math.ceil(num_inference_steps / strength),
        #         guidance_scale = guidance_scale,
        #         strength = .18,
        #         generator = _generator
        #     ).images[0]

        output_image.paste(input_image.convert('RGBA'), (0, 0), input_image.convert('RGBA'))

        # # output
        output_url = extract_origin_pathname(upload_url)
        output = { 'output_url': output_url }

        if debug:
            output_image.save('sample.png')
        else:
            upload_image(upload_url, output_image)

        return output
    # caught http[s] error
    except requests.exceptions.RequestException as e:
        return { 'error': e.args[0] }

runpod.serverless.start({ 'handler': run })
