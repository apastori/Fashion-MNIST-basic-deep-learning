"""Provide the sigmoid activation operation."""

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.asarray import as_array


class Sigmoid(Operation):
    """Sigmoid activation function."""

    def _output(self) -> np.ndarray:
        """Forward Pass during Prediction."""
        ## The output value of this function is stored in output through forward function in Operation class
        return as_array(1 / (1 + np.exp(self._require_input() * -1)))

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Backward Pass during Training."""
        output = self._require_output()
        sigmoid_derivative = output * (1 - output)
        return as_array(grad * sigmoid_derivative)
