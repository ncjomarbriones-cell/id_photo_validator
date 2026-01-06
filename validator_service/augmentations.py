import random
from typing import Iterable, List

from PIL import Image, ImageEnhance, ImageFilter


def _motion_blur(img: Image.Image, radius: float = 5.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _color_cast(img: Image.Image, factor: float = 1.25) -> Image.Image:
    r, g, b = img.split()
    r = r.point(lambda x: min(255, x * factor))
    return Image.merge("RGB", (r, g, b))


def _add_noise(img: Image.Image, std: float = 12.0) -> Image.Image:
    import numpy as np

    arr = np.asarray(img).astype("float32")
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def generate_bad_variants(image: Image.Image, max_variants: int = 8) -> List[Image.Image]:
    """
    Produce degraded copies of a clean ID photo.
    Kept small to avoid flooding the dataset.
    """
    variants: List[Image.Image] = []

    # Blur (Gaussian + motion-like)
    variants.append(image.filter(ImageFilter.GaussianBlur(radius=2.5)))
    variants.append(image.filter(ImageFilter.GaussianBlur(radius=4.0)))
    variants.append(_motion_blur(image, radius=6.0))

    # Dark / bright / low contrast
    variants.append(ImageEnhance.Brightness(image).enhance(0.5))
    variants.append(ImageEnhance.Brightness(image).enhance(1.6))
    variants.append(ImageEnhance.Contrast(image).enhance(0.6))

    # Color cast / glare-ish
    variants.append(_color_cast(image, factor=1.35))

    # Tilt
    variants.append(image.rotate(16, expand=True, fillcolor=(255, 255, 255)))
    variants.append(image.rotate(-16, expand=True, fillcolor=(255, 255, 255)))

    # Crop forehead/chin / off-center crop
    w, h = image.size
    chop = int(0.1 * h)
    variants.append(image.crop((0, chop, w, h)))  # remove forehead
    variants.append(image.crop((0, 0, w, h - chop)))  # remove chin
    variants.append(image.crop((int(0.05 * w), 0, w, h)))  # left shift

    # Noise
    variants.append(_add_noise(image, std=10.0))

    random.shuffle(variants)
    return variants[:max_variants]


def generate_good_variants(image: Image.Image, max_variants: int = 2) -> List[Image.Image]:
    """
    Mild perturbations that should keep the image acceptable.
    """
    variants: List[Image.Image] = []
    variants.append(ImageEnhance.Brightness(image).enhance(1.05))
    variants.append(ImageEnhance.Contrast(image).enhance(1.05))
    random.shuffle(variants)
    return variants[:max_variants]
