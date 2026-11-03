"""Provide a base implementation for sequential neural-network layers."""

from abc import abstractmethod

import numpy as np

from mnistnn.layer_parent import LayerParent
from mnistnn.operation import Operation
from mnistnn.operation_param import ParamOperation
from mnistnn.utils.shape import assert_same_shape


class GenericLayer(LayerParent):
    """A "layer" of neurons in a neural network."""

    def __init__(self, neurons: int) -> None:
        """Initialize the layer with the specified number of neurons.

        The number of "neurons" roughly corresponds to the "breadth" of the
        layer.
        """
        self.neurons: int = neurons
        self.first: bool = True
        self.params: list[np.ndarray] = []
        self.param_grads: list[np.ndarray] = []
        self.operations: list[Operation] = []
        self.input_: np.ndarray | None = None
        self.output: np.ndarray | None = None

    @abstractmethod
    def _setup_layer(self, input_: np.ndarray) -> None:
        """Set up the layer from the first forward-pass input."""
        raise NotImplementedError()

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Pass input forward through a series of operations."""
        if self.first:
            self._setup_layer(input_)
            self.first = False
        self.input_ = input_
        current: np.ndarray = input_
        for operation in self.operations:
            current = operation.forward(current)
        self.output = current
        return current

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Pass output_grad backward through a series of operations.

        Checks appropriate shapes.
        """
        assert_same_shape(self.output, output_grad)
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
            input_grad = output_grad
        self.get_grads()
        return input_grad

    def get_grads(self) -> tuple[np.ndarray, ...]:
        """Extract the _param_grads from the layer's operations."""
        self.param_grads = []
        for operation in self.operations:
            if isinstance(operation, ParamOperation):
                if operation.param_grad is None:
                    raise RuntimeError("Parameter gradient is not available.")
                self.param_grads.append(operation.param_grad)
        return tuple(self.param_grads)

    def get_params(self) -> tuple[np.ndarray, ...]:
        """Extract the _params from the layer's operations."""
        self.params = []
        for operation in self.operations:
            if isinstance(operation, ParamOperation):
                self.params.append(operation.param)
        return tuple(self.params)
