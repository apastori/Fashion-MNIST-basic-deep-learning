"""Provide the Fashion-MNIST neural-network implementation."""

import numpy as np

from mnistnn.functions.CrossEntropyLoss import CrossEntropyLoss
from mnistnn.functions.ReLU import ReLU
from mnistnn.functions.SoftmaxStable import SoftmaxStable
from mnistnn.layer import Layer
from mnistnn.layer_parent import LayerParent
from mnistnn.loss import Loss
from mnistnn.utils import horizontal_flip_images


class MNISTNeuralNetwork:
    """Neural network for Fashion-MNIST clothing classification.

    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden Layer 1: 512 units with ReLU and dropout 0.1
    - Hidden Layer 2: 256 units with ReLU and dropout 0.2
    - Hidden Layer 3: 128 units with ReLU and dropout 0.2
    - Output Layer: 10 units with Softmax
    """

    def __init__(self) -> None:
        """Initialize the network layers and loss function."""
        # Initialize layers
        self.layers: list[LayerParent] = [
            Layer(
                input_size=784,
                output_size=512,
                activation=ReLU(),
                dropout_rate=0.1,
            ),
            Layer(
                input_size=512,
                output_size=256,
                activation=ReLU(),
                dropout_rate=0.2,
            ),
            Layer(
                input_size=256,
                output_size=128,
                activation=ReLU(),
                dropout_rate=0.2,
            ),
            Layer(input_size=128, output_size=10, activation=SoftmaxStable()),
        ]
        # Loss function
        self.loss_fn: Loss = CrossEntropyLoss()
        # SGD momentum buffers, allocated on the first parameter update
        self._weight_velocities: list[np.ndarray] | None = None
        self._bias_velocities: list[np.ndarray] | None = None

    def _set_training(self, training: bool) -> None:
        """Enable or disable dropout on every dense layer."""
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.training = training

    def forward(self, input_: np.ndarray, training: bool = True) -> np.ndarray:
        """Perform a forward pass through the network."""
        self._set_training(training)
        # Flatten input if needed (batch_size, 28, 28) -> (batch_size, 784)
        if len(input_.shape) == 3:
            if input_.shape[1:] != (28, 28):
                raise ValueError(
                    f"Expected images with shape (batch_size, 28, 28), "
                    f"got {input_.shape}"
                )
            input_ = input_.reshape(input_.shape[0], -1)

        x: np.ndarray = input_
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def _backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Perform a backward pass through the network."""
        input_grad: np.ndarray = output_grad

        for layer in reversed(self.layers):
            input_grad = layer.backward(input_grad)

        return input_grad

    def _compute_loss(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute the cross-entropy loss."""
        return self.loss_fn.forward(predictions, targets)

    def _update_parameters(
        self,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        """Update weights and biases using SGD with momentum and L2 decay."""
        if self._weight_velocities is None or self._bias_velocities is None:
            self._weight_velocities = []
            self._bias_velocities = []
            for layer in self.layers:
                init_weights: np.ndarray
                init_bias: np.ndarray
                init_weights, init_bias = layer.get_params()
                self._weight_velocities.append(np.zeros_like(init_weights))
                self._bias_velocities.append(np.zeros_like(init_bias))

        for index, layer in enumerate(self.layers):
            weights: np.ndarray
            bias: np.ndarray
            weights, bias = layer.get_params()
            grad_weights: np.ndarray
            grad_bias: np.ndarray
            grad_weights, grad_bias = layer.get_grads()
            weight_velocity: np.ndarray
            bias_velocity: np.ndarray
            weight_velocity = (
                momentum * self._weight_velocities[index] + grad_weights
            )
            bias_velocity = momentum * self._bias_velocities[index] + grad_bias
            self._weight_velocities[index] = weight_velocity
            self._bias_velocities[index] = bias_velocity
            weights -= learning_rate * (
                weight_velocity + weight_decay * weights
            )
            bias -= learning_rate * bias_velocity

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> float:
        """Perform a single training step."""
        # Forward pass
        predictions: np.ndarray = self.forward(x, training=True)

        # Compute loss
        loss: float = self._compute_loss(predictions, y)

        # Backward pass
        grad: np.ndarray = self.loss_fn.backward()
        self._backward(grad)

        # Update parameters
        self._update_parameters(learning_rate, momentum, weight_decay)

        return loss

    def predict(self, x: np.ndarray, tta_hflip: bool = True) -> np.ndarray:
        """Predict class labels, optionally averaging a horizontal flip."""
        probabilities: np.ndarray = self.forward(x, training=False)
        if tta_hflip:
            flipped_images: np.ndarray = horizontal_flip_images(x)
            flipped_probabilities: np.ndarray = self.forward(
                flipped_images, training=False
            )
            probabilities = (probabilities + flipped_probabilities) * 0.5
        return np.asarray(np.argmax(probabilities, axis=1))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy on a dataset."""
        predictions: np.ndarray = self.predict(x)

        # Convert one-hot encoded labels to class indices if needed
        labels: np.ndarray = y
        if len(y.shape) == 2:
            labels = np.asarray(np.argmax(y, axis=1))

        accuracy: float = float(np.mean(predictions == labels))
        return accuracy
