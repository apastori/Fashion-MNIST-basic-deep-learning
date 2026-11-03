"""Provide plotting helpers for Fashion-MNIST images and statistics."""

from typing import NoReturn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class MNISTVisualizer:
    """Utility class for visualizing Fashion-MNIST data."""

    def __new__(cls) -> NoReturn:
        """Prevent instantiation of this static utility class."""
        raise TypeError("MNISTVisualizer is a static utility class")

    def __init__(self) -> None:
        """Reject direct initialization of this static utility class."""
        raise TypeError("MNISTVisualizer is a static utility class")

    @staticmethod
    def plot_single_image(
        image: np.ndarray,
        label: int,
        title: str = "Fashion-MNIST Image",
    ) -> None:
        """Plot a single Fashion-MNIST image with its label."""
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap="gray")
        plt.title(f"{title} - Label: {label}")
        plt.axis("off")
        plt.show()

    @staticmethod
    def plot_grid(
        images: np.ndarray,
        labels: np.ndarray,
        num_images: int = 9,
        title: str = "Fashion-MNIST Images Grid",
    ) -> None:
        """Plot multiple Fashion-MNIST images in a grid."""
        display_count: int = min(num_images, len(images))
        grid_size: int = int(np.ceil(np.sqrt(display_count)))
        fig: matplotlib.figure.Figure
        axes: np.ndarray
        flat_axes: np.ndarray
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        flat_axes = np.asarray(axes).reshape(-1)

        for i, ax in enumerate(flat_axes):
            if i < display_count:
                ax.imshow(images[i], cmap="gray")
                ax.set_title(f"Label: {labels[i]}")
                ax.axis("off")
            else:
                ax.axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_image_statistics(
        images: np.ndarray,
        title: str = "Fashion-MNIST Image Statistics",
    ) -> None:
        """Plot pixel intensity distribution."""
        pixels: np.ndarray = images.flatten()
        plt.figure(figsize=(10, 5))
        plt.hist(pixels, bins=256, color="blue", alpha=0.7)
        plt.title(title)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.show()
