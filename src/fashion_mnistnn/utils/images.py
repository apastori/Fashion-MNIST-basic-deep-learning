"""Provide image-array helpers for Fashion-MNIST-shaped (28x28) batches."""

import numpy as np


def horizontal_flip_images(images: np.ndarray) -> np.ndarray:
    """Flip 28x28 images left-to-right, keeping the input layout."""
    if images.ndim == 3:
        if images.shape[1:] != (28, 28):
            raise ValueError(
                f'Expected images with shape (batch_size, 28, 28), '
                f'got {images.shape}'
            )
        return np.ascontiguousarray(images[:, :, ::-1])
    if images.ndim == 2 and images.shape[1] == 784:
        flipped: np.ndarray = images.reshape(-1, 28, 28)[:, :, ::-1]
        return np.ascontiguousarray(flipped).reshape(-1, 784)
    raise ValueError(
        f'Images to flip must have shape (N, 28, 28) or (N, 784), '
        f'got {images.shape}'
    )
