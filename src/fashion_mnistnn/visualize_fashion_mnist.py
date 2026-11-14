"""Visualize the Fashion-MNIST dataset."""

import numpy as np


def visualize_fashion_mnist(
    train_images: np.ndarray, train_labels: np.ndarray, visualize: bool = True
) -> None:
    """Visualize Fashion-MNIST training images when ``visualize`` is True."""
    if visualize:
        from fashion_mnistnn.fashion_mnist_visualizer import (
            FashionMNISTVisualizer,
        )

        FashionMNISTVisualizer.plot_single_image(
            train_images[0], int(train_labels[0])
        )
        FashionMNISTVisualizer.plot_grid(
            train_images[:9], train_labels[:9], num_images=9
        )
        FashionMNISTVisualizer.plot_image_statistics(train_images)


__all__ = ['visualize_fashion_mnist']
