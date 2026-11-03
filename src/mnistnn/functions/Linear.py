"""Provide a fully connected linear transformation."""

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.asarray import as_array


class Linear(Operation):
    """Linear transformation (fully connected layer)."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialize weights and bias for the linear transformation."""
        super().__init__()

        self.weights: np.ndarray = np.random.randn(
            input_size, output_size
        ) * np.sqrt(2.0 / input_size)
        self.bias: np.ndarray = np.zeros((1, output_size))

    """Forward Pass during Prediction"""

    def _output(self) -> np.ndarray:
        ## output value of this function is stored in output through forward function in Operation class
        return as_array(
            np.dot(self._require_input(), self.weights) + self.bias
        )

    """Backward Pass during Training"""

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        self.grad_weights: np.ndarray = as_array(
            np.dot(self._require_input().T, grad)
        )
        self.grad_bias: np.ndarray = np.sum(grad, axis=0, keepdims=True)
        return as_array(np.dot(grad, self.weights.T))
