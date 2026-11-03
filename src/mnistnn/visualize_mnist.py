"""Visualize the Fashion-MNIST dataset."""

import numpy as np


def visualize_mnist(
    train_images: np.ndarray, train_labels: np.ndarray, visualize: bool = True
) -> None:
    """Visualize Fashion-MNIST training images when ``visualize`` is True."""
    if visualize:
        from mnistnn.mnist_visualizer import MNISTVisualizer

        MNISTVisualizer.plot_single_image(
            train_images[0], int(train_labels[0])
        )
        MNISTVisualizer.plot_grid(
            train_images[:9], train_labels[:9], num_images=9
        )
        MNISTVisualizer.plot_image_statistics(train_images)


__all__ = ['visualize_mnist']
