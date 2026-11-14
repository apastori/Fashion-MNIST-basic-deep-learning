"""Define the abstract interface for neural-network layers."""

from abc import ABC, abstractmethod

import numpy as np


class LayerParent(ABC):
    """Abstract base class for layers."""

    @abstractmethod
    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Compute the layer output for the given input."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Propagate output gradients backward through the layer."""
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> tuple[np.ndarray, ...]:
        """Return the parameters used by the layer."""
        raise NotImplementedError

    @abstractmethod
    def get_grads(self) -> tuple[np.ndarray, ...]:
        """Return gradients for the layer parameters."""
        raise NotImplementedError
