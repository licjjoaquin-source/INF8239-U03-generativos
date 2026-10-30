from __future__ import annotations

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    return (images.astype("float32") / 255.0)[..., None]


def validate_images(images: np.ndarray) -> None:
    if images.ndim != 4 or images.shape[1:] != (28, 28, 1):
        raise ValueError(f"Forma inesperada: {images.shape}")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Las imágenes deben contener valores finitos entre 0 y 1")
