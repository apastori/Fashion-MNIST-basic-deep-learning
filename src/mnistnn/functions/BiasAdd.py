"""Provide the bias-addition operation."""

import numpy as np

from mnistnn.operation_param import ParamOperation
from mnistnn.utils.asarray import as_array


class BiasAdd(ParamOperation):
    """Compute bias addition."""

    def __init__(self, bias: np.ndarray) -> None:
        """Initialize the operation with self.param = bias.

        Check appropriate shape.
        """
        assert bias.shape[0] == 1
        super().__init__(bias)

    def _output(self) -> np.ndarray:
        """Compute Output."""
        return as_array(self._require_input() + self.param)

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Compute input gradient."""
        return as_array(np.ones_like(self._require_input()) * grad)

    def _param_grad(self, grad: np.ndarray) -> np.ndarray:
        """Compute parameter gradient."""
        param_grad: np.ndarray = np.ones_like(self.param) * grad
        return as_array(
            np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
        )
