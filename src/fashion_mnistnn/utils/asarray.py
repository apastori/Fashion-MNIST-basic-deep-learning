"""Provide helpers that convert values to NumPy arrays."""

import numpy as np


def as_array(value: object) -> np.ndarray:
    """Return ``value`` as a NumPy array."""
    return np.asarray(value)
