"""Image processing utilities."""
import numpy as np
from typing import Tuple

def resize(image: np.ndarray, size: Tuple[int, int], method: str = "bilinear") -> np.ndarray:
    h, w = image.shape[:2]
    new_h, new_w = size
    if method == "nearest":
        row_idx = (np.arange(new_h) * h / new_h).astype(int).clip(0, h-1)
        col_idx = (np.arange(new_w) * w / new_w).astype(int).clip(0, w-1)
        return image[np.ix_(row_idx, col_idx)]
    return image  # bilinear placeholder

def normalize(image: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    return (image - mean) / std

def denormalize(image: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    return image * std + mean

def center_crop(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    new_h, new_w = size
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return image[top:top+new_h, left:left+new_w]

def pad_to_multiple(image: np.ndarray, multiple: int = 8) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    padded = np.zeros((new_h, new_w) + image.shape[2:], dtype=image.dtype)
    padded[:h, :w] = image
    return padded
