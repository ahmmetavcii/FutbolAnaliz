"""Small, dependency-light image transforms for player crops."""

from __future__ import annotations

import random

import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class JerseyTransform:
    """Resize, mildly augment, and normalize a crop.

    Augmentation uses Python's worker-seeded RNG and is intentionally conservative:
    horizontal flips preserve jersey digits while aggressive geometric changes do not.
    """

    def __init__(self, image_size: tuple[int, int] = (128, 64), training: bool = False) -> None:
        self.height, self.width = image_size
        self.training = training

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.training:
            if random.random() < 0.5:
                image = F.hflip(image)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        image = F.resize(image, [self.height, self.width], antialias=True)
        return F.normalize(F.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


def build_transform(
    image_size: tuple[int, int] | list[int] = (128, 64), *, training: bool = False
) -> JerseyTransform:
    return JerseyTransform((int(image_size[0]), int(image_size[1])), training=training)
