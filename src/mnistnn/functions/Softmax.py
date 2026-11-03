"""Provide the softmax activation operation."""

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.asarray import as_array


class Softmax(Operation):
    """Softmax activation function."""

    """Forward Pass during Prediction"""

    def _output(self) -> np.ndarray:
        # Check if the input is a 2D array
        if self.input_ is None or self.input_.ndim != 2:
            raise ValueError("Input to Softmax must be a 2D array")
        exp_input = np.exp(self.input_)
        # Output already saved in self.output through forward function in Operation class
        return as_array(exp_input / np.sum(exp_input, axis=1, keepdims=True))

    """Backward Pass during Training"""

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Compute the gradient with respect to the input logits."""
        if self.output is None:
            raise RuntimeError(
                "Softmax backward requires a previous forward pass"
            )
        probabilities = self.output

        return as_array(
            probabilities
            * (grad - np.sum(grad * probabilities, axis=1, keepdims=True))
        )
