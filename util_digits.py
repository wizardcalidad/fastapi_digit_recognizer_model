
from numpy import asarray
from PIL import Image


def convert_image_to_array(file, height, width, depth):
    with file as image:
        image = Image.open(image)
        image = image.resize((height, width), Image.ANTIALIAS)
        return asarray(image).reshape((1, height, width, depth))