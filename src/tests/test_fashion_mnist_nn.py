"""Tests for the Fashion-MNIST neural network class."""

from pathlib import Path

import numpy as np
import pytest

from fashion_mnistnn.fashion_mnist_neural_network import (
    FashionMNISTNeuralNetwork,
)


class TestFashionMNISTNeuralNetwork:
    """Test suite for the Fashion-MNIST neural network class."""

    def test_save_and_load_weights_round_trip(self, tmp_path: Path) -> None:
        """Saved weights must restore identical forward outputs."""
        rng: np.random.Generator = np.random.default_rng(0)
        source: FashionMNISTNeuralNetwork = FashionMNISTNeuralNetwork()
        destination: FashionMNISTNeuralNetwork = FashionMNISTNeuralNetwork()
        batch: np.ndarray = rng.normal(size=(8, 784))

        weight_path: Path = tmp_path / 'fashion_mnist.npz'
        source.save_weights(weight_path)
        destination.load_weights(weight_path)

        source_probs: np.ndarray = source.forward(batch, training=False)
        destination_probs: np.ndarray = destination.forward(
            batch, training=False
        )
        assert np.allclose(source_probs, destination_probs)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
