"""Tests for Fashion-MNIST visualizer class."""

import numpy as np
import pytest

from fashion_mnistnn.data_config import dict_config_test, dict_config_train
from fashion_mnistnn.fashion_mnist_data_loader import (
    FashionMNISTDataLoader,
    FashionMNISTImagesInfo,
    FashionMNISTLabelsInfo,
)
from fashion_mnistnn.fashion_mnist_visualizer import FashionMNISTVisualizer


class TestFashionMNISTVisualizer:
    """Test suite for Fashion-MNIST visualizer class."""

    @pytest.fixture
    def data_loader(self) -> FashionMNISTDataLoader:
        """Initialize Fashion-MNIST data loader with configuration."""
        loader: FashionMNISTDataLoader = FashionMNISTDataLoader(
            dict_config_train, dict_config_test
        )
        return loader

    @pytest.fixture
    def training_data(
        self, data_loader: FashionMNISTDataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve training images and labels."""
        train_images: FashionMNISTImagesInfo
        train_labels: FashionMNISTLabelsInfo
        train_images, train_labels = data_loader.training_info
        images_array: np.ndarray = train_images["images"]
        labels_array: np.ndarray = train_labels["labels"]
        return images_array, labels_array

    def test_plot_single_image(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that plotting a single image does not raise errors."""
        images_array: np.ndarray
        labels_array: np.ndarray
        images_array, labels_array = training_data
        try:
            FashionMNISTVisualizer.plot_single_image(images_array[0], labels_array[0])
        except Exception as e:
            raise AssertionError(
                f"Plotting single image raised an exception: {e}"
            ) from e

    def test_plot_random_images_grid(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that plotting a grid of 20 random Fashion-MNIST images works."""
        images_array: np.ndarray
        labels_array: np.ndarray
        images_array, labels_array = training_data
        rng: np.random.Generator = np.random.default_rng(seed=42)
        indices: np.ndarray = rng.choice(
            len(images_array), size=20, replace=False
        )

        random_images: np.ndarray = images_array[indices]
        random_labels: np.ndarray = labels_array[indices]

        try:
            FashionMNISTVisualizer.plot_grid(
                random_images,
                random_labels,
                num_images=20,
                title="20 Random Fashion-MNIST Samples",
            )
        except Exception as e:
            raise AssertionError(
                'Plotting 20 random Fashion-MNIST images raised an '
                f'exception: {e}'
            ) from e
