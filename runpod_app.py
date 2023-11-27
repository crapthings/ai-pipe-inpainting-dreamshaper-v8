import math
import requests

import numpy as np
import torch
import runpod

from utils import extract_origin_pathname, upload_image, rounded_size
from inpainting import inpainting, img2img

def run (job, _generator = None):
    # prepare task
    try:
        print('debug', job)

        _input = job.get('input')

        debug = _input.get('debug')
        upload_url = _input.get('upload_url')

        prompt = _input.get('prompt', 'a dog')
        negative_prompt = _input.get('negative_prompt', '')
        width = int(np.clip(_input.get('width', 768), 256, 1024))
        height = int(np.clip(_input.get('height', 768), 256, 1024))
        num_inference_steps = int(np.clip(_input.get('num_inference_steps', 30), 20, 150))
        guidance_scale = float(np.clip(_input.get('guidance_scale', 13.0), 0, 30))
        seed = _input.get('seed')

        upscale = _input.get('upscale')

        strength = _input.get('strength')

        if strength is not None:
            strength = float(np.clip(strength, 0, 1))

        if upscale is not None:
            upscale = float(np.clip(upscale, 1, 4))

        renderWidth, renderHeight = rounded_size(width, height)

        if seed is not None:
            _generator = torch.Generator(device = 'cuda').manual_seed(seed)

        output_image = txt2img(
            prompt = prompt,
            negative_prompt = negative_prompt,
            width = renderWidth,
            height = renderHeight,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            generator = _generator
        ).images[0]

        output_image = output_image.resize([width, height])

        if upscale is not None:
            output_image = output_image.resize([int(width * upscale), int(height * upscale)])

        if strength is not None:
            output_image = img2img(
                image = output_image,
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps = math.ceil(num_inference_steps / strength),
                guidance_scale = guidance_scale,
                strength = strength,
                generator = _generator
            ).images[0]

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
