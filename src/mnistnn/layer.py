"""Provide a layer combining a linear transformation and activation."""

import numpy as np

from mnistnn.functions.Dropout import Dropout
from mnistnn.functions.Linear import Linear
from mnistnn.layer_parent import LayerParent
from mnistnn.operation import Operation


class Layer(LayerParent):
    """Layer combining linear transformation, activation, and optional dropout."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Operation | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the layer with dimensions, activation, and dropout."""
        if (input_size <= 0) or (output_size <= 0):
            raise ValueError(
                "Input and output sizes must be positive integers."
            )
        if activation is None:
            raise ValueError("Activation function must be provided.")
        if dropout_rate < 0.0 or dropout_rate >= 1.0:
            raise ValueError('Dropout rate must be in the range [0, 1).')
        self.linear: Linear = Linear(input_size, output_size)
        self.activation: Operation | None = activation
        self.dropout: Dropout | None = (
            Dropout(dropout_rate) if dropout_rate > 0.0 else None
        )
        self.training: bool = True
        # Store the output of the linear layer for backward pass
        self.linear_output: np.ndarray | None = None

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Compute the layer output for the given input."""
        self.linear_output = self.linear.forward(input_)
        output: np.ndarray = self.linear_output
        if self.activation:
            output = self.activation.forward(output)
        if self.dropout is not None:
            self.dropout.training = self.training
            output = self.dropout.forward(output)
        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Propagate gradients backward through the layer."""
        output_grad_final: np.ndarray = output_grad
        if self.dropout is not None:
            output_grad_final = self.dropout.backward(output_grad_final)
        if self.activation:
            output_grad_final = self.activation.backward(output_grad_final)
        return self.linear.backward(output_grad_final)

    def get_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the layer weights and bias."""
        return self.linear.weights, self.linear.bias

    def get_grads(self) -> tuple[np.ndarray, np.ndarray]:
        """Return gradients for the layer weights and bias."""
        return self.linear.grad_weights, self.linear.grad_bias
