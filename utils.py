import requests
from io import BytesIO
from urllib.parse import urlparse
from PIL import Image

def buff_png (image):
    buff = BytesIO()
    image.save(buff, format = 'PNG')
    buff.seek(0)
    return buff

def upload_image (url, image):
    response = requests.put(url, data = buff_png(image), headers = { 'Content-Type': 'image/png' })
    response.raise_for_status()

def extract_origin_pathname (url):
    parsed_url = urlparse(url)
    origin_pathname = parsed_url.scheme + '://' + parsed_url.netloc + parsed_url.path
    return origin_pathname

def rounded_size (width, height):
    rounded_width = (width // 8) * 8
    rounded_height = (height // 8) * 8

    if width % 8 >= 4:
        rounded_width += 8
    if height % 8 >= 4:
        rounded_height += 8

    return int(rounded_width), int(rounded_height)

def zoom_and_crop (image, zoom_factor):
    new_width = int(image.width * zoom_factor)
    new_height = int(image.height * zoom_factor)

    image = image.resize((new_width, new_height))

    left = (new_width - image.width) // 2
    top = (new_height - image.height) // 2
    right = left + image.width
    bottom = top + image.height

    image = image.crop((left, top, right, bottom))

    return image

def open_url (url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    output_image = Image.open(image_data)
    return output_image

def resize_image (original_image, new_width):
    aspect_ratio = float(new_width) / original_image.size[0]
    new_height = int(aspect_ratio * original_image.size[1])
    resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

def sc(self, clip_input, images): return images, [False for i in images]
