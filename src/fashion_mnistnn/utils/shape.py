"""Provide array-shape validation helpers."""

import numpy as np


def assert_same_shape(x: np.ndarray | None, y: np.ndarray | None) -> None:
    """Raise an error if two arrays are missing or do not match in shape."""
    if x is None or y is None:
        raise RuntimeError("Expected arrays after a forward pass")
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: expected {x.shape}, got {y.shape}")
