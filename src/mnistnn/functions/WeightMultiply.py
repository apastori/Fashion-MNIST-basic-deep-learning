"""Provide the parameterized weight-multiplication operation."""

import numpy as np

from mnistnn.operation_param import ParamOperation
from mnistnn.utils.asarray import as_array


class WeightMultiply(ParamOperation):
    """Weight multiplication operation for a neural network."""

    def __init__(self, weights: np.ndarray) -> None:
        """Initialize the operation with the supplied weights."""
        super().__init__(weights)

    def _output(self) -> np.ndarray:
        """Compute output."""
        return as_array(np.dot(self._require_input(), self.param))

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Compute input gradient."""
        return as_array(np.dot(grad, np.transpose(self.param, (1, 0))))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """Compute parameter gradient."""
        return as_array(
            np.dot(np.transpose(self._require_input(), (1, 0)), output_grad)
        )
