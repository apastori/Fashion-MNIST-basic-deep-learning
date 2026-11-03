"""Provide the numerically stable softmax activation operation."""

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.asarray import as_array


class SoftmaxStable(Operation):
    """Compute softmax after subtracting the maximum input value."""

    """Forward Pass during Prediction"""

    def _output(self) -> np.ndarray:
        # Check if the input is a 2D array
        if self.input_ is None or self.input_.ndim != 2:
            raise ValueError("Input to SoftmaxStable must be a 2D array")

        # Subtract the maximum value in each row for numerical stability
        shifted_input = self.input_ - np.max(
            self.input_,
            axis=1,
            keepdims=True,
        )

        exp_input = np.exp(shifted_input)

        return as_array(
            exp_input
            / np.sum(
                exp_input,
                axis=1,
                keepdims=True,
            )
        )

    """Backward Pass during Training"""

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError(
                "SoftmaxStable backward requires a previous forward pass"
            )

        probabilities = self.output

        return as_array(
            probabilities
            * (
                grad
                - np.sum(
                    grad * probabilities,
                    axis=1,
                    keepdims=True,
                )
            )
        )
