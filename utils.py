import requests
from io import BytesIO
from urllib.parse import urlparse

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
