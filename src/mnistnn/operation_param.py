"""Define operations that calculate gradients for trainable parameters."""

from abc import abstractmethod

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.shape import assert_same_shape


class ParamOperation(Operation):
    """An Operation with parameters."""

    def __init__(self, param: np.ndarray) -> None:
        """Initialize the parameterized operation."""
        super().__init__()
        self.param: np.ndarray = param
        self.param_grad: np.ndarray | None = None

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Call self._input_grad and self._param_grad.

        Checks appropriate shapes.
        Input grad shape must match input shape.
        Param grad shape must match param shape.
        """
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)
        return self.input_grad

    @abstractmethod
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """Every subclass of ParamOperation must implement _param_grad."""
        raise NotImplementedError()
