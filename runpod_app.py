import math
import requests

import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers.utils import load_image
import runpod

from utils import extract_origin_pathname, upload_image, rounded_size, open_url, resize_image
from inpainting import inpainting, compel

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
        num_inference_steps = int(np.clip(_input.get('num_inference_steps', 30), 20, 150))
        guidance_scale = float(np.clip(_input.get('guidance_scale', 13.0), 0, 30))
        strength = _input.get('strength', 1)
        seed = _input.get('seed')
        invert = _input.get('invert')
        fix = _input.get('fix')

        limit = _input.get('limit', 768)

        if strength is not None:
            strength = float(np.clip(strength, 0, 1))

        input_image = open_url(input_url)
        mask_image = load_image(mask_url)

        input_image = resize_image(input_image, limit)
        mask_image = resize_image(mask_image, limit)

        if invert is True:
            mask_image = ImageOps.invert(mask_image)

        renderWidth, renderHeight = rounded_size(input_image.width, input_image.height)

        if seed is not None:
            _generator = torch.Generator(device = 'cuda').manual_seed(seed)

        prompt_embeds = compel.build_conditioning_tensor(prompt)
        negative_prompt_embeds = compel.build_conditioning_tensor(negative_prompt)

        output_image = inpainting(
            image = input_image.convert('RGB'),
            mask_image = mask_image,
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            width = renderWidth,
            height = renderHeight,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            strength = strength,
            generator = _generator
        ).images[0]

        output_image = output_image.resize(input_image.size)

        if fix is not True:
            #
            mask_image_arr = np.array(mask_image.convert('L'))
            mask_image_arr = mask_image_arr[:, :, None]
            mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
            mask_image_arr[mask_image_arr < 0.5] = 0
            mask_image_arr[mask_image_arr >= 0.5] = 1
            unmasked_unchanged_image_arr = (1 - mask_image_arr) * input_image + mask_image_arr * output_image
            output_image = Image.fromarray(unmasked_unchanged_image_arr.round().astype('uint8'))

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
