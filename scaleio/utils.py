from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image


class UnsupportedScaleError(ValueError):
    pass


class ImageLoadError(Exception):
    pass


SUPPORTED_SCALES = (2, 4, 8)


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def validate_scale(scale: int) -> None:
    if scale not in SUPPORTED_SCALES:
        raise UnsupportedScaleError(f"Scale must be one of {SUPPORTED_SCALES}, got {scale}")


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(input_: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(input_, (str, Path)):
        path = Path(input_)
        if not path.exists():
            raise ImageLoadError(f"Image file not found: {path}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise ImageLoadError(f"Failed to load image: {e}")
    elif isinstance(input_, Image.Image):
        return input_.convert("RGB")
    elif isinstance(input_, np.ndarray):
        return Image.fromarray(input_)
    else:
        raise ImageLoadError(
            f"Unsupported input type: {type(input_)}. Expected str, Path, PIL.Image, or np.ndarray"
        )
