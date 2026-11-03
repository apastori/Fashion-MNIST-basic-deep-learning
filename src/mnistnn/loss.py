"""Define the abstract interface for neural-network loss functions."""

from abc import ABC, abstractmethod

import numpy as np

from mnistnn.utils.shape import assert_same_shape


class Loss(ABC):
    """Abstract base class for loss functions."""

    def __init__(self) -> None:
        """Initialize the loss state before a forward pass."""
        self.prediction: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.input_grad: np.ndarray | None = None

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute the actual loss value."""
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        loss_value = self._output()
        return loss_value

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss value.

        Return the gradient with respect to the loss function input.
        """
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    @abstractmethod
    def _output(self) -> float:
        """Every subclass of "Loss" must implement the _output function."""
        raise NotImplementedError()

    @abstractmethod
    def _input_grad(self) -> np.ndarray:
        """Every subclass of "Loss" must implement the _input_grad function."""
        raise NotImplementedError()
