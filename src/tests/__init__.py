"""Define the test package for the Fashion-MNIST project."""

import os

import pytest

"""Initialization for the tests package."""

if __name__ == "__main__":
    test_directory: str = os.path.dirname(__file__)
    pytest.main([os.path.join(test_directory, "test_mnist_data_loader.py")])
    pytest.main([os.path.join(test_directory, "test_visualize_data.py")])
    pytest.main([os.path.join(test_directory, "test_mnist_nn.py")])
