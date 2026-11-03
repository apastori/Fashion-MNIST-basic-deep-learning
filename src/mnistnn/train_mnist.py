"""Main package to expose the different components."""

from __future__ import annotations

import logging
import math
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mnistnn.log_config import configure_file_logging
from mnistnn.mnist_data_loader import (
    MNISTDataLoader as MNISTDataLoader,
)
from mnistnn.mnist_data_loader import (
    MNISTImagesInfo,
    MNISTLabelsInfo,
)
from mnistnn.visualize_mnist import visualize_mnist

if TYPE_CHECKING:
    from mnistnn.mnist_neural_network import MNISTNeuralNetwork

logger = logging.getLogger(__name__)

LEARNING_RATE_FLOOR: float = 1e-4
HORIZONTAL_FLIP_PROBABILITY: float = 0.5


def _prepare_images(images: np.ndarray, dataset_name: str) -> np.ndarray:
    """Validate Fashion-MNIST image dimensions and flatten image batches."""
    if images.ndim == 3:
        if images.shape[1:] != (28, 28):
            raise ValueError(
                f'{dataset_name} images must have shape (N, 28, 28), '
                f'got {images.shape}'
            )
        return images.reshape(images.shape[0], -1)
    if images.ndim == 2 and images.shape[1] == 784:
        return images
    raise ValueError(
        f'{dataset_name} images must have shape (N, 28, 28) or (N, 784), '
        f'got {images.shape}'
    )


def _normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale pixels to [0, 1] and mean-center them to [-0.5, 0.5]."""
    return images.astype(np.float64) / 255.0 - 0.5


def _cosine_learning_rate(
    initial_rate: float, floor: float, epoch: int, epochs: int
) -> float:
    """Return the cosine-annealed learning rate for a 0-based epoch index."""
    if epochs <= 1:
        return initial_rate
    progress: float = epoch / (epochs - 1)
    cosine: float = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (initial_rate - floor) * cosine


def _random_horizontal_flip(
    images: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomly flip flattened 28x28 images left-to-right during training."""
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(
            f'Images to flip must have shape (N, 784), got {images.shape}'
        )
    batch_images: np.ndarray = images.reshape(-1, 28, 28)
    flip_mask: npt.NDArray[np.bool_] = (
        rng.random(len(batch_images)) < HORIZONTAL_FLIP_PROBABILITY
    )
    if np.any(flip_mask):
        batch_images[flip_mask] = np.ascontiguousarray(
            batch_images[flip_mask, :, ::-1]
        )
    return batch_images.reshape(-1, 784)


def train_mnist(
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 0.03,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    visualize: bool = True,
    append_logs: bool = True,
) -> MNISTNeuralNetwork:
    """Load Fashion-MNIST and train the network, optionally starting a fresh log."""
    if epochs <= 0:
        raise ValueError('epochs must be greater than zero')
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be greater than zero')
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError('momentum must be in the range [0, 1)')
    if weight_decay < 0.0:
        raise ValueError('weight_decay must be greater than or equal to zero')

    log_path = configure_file_logging(logger, append=append_logs)
    logger.info('Writing Fashion-MNIST training logs to %s', log_path)
    logger.info(
        'Starting Fashion-MNIST training (epochs=%d, batch_size=%d, '
        'learning_rate=%g, momentum=%g, weight_decay=%g, '
        'schedule=cosine, tta_hflip=True, visualize=%s)',
        epochs,
        batch_size,
        learning_rate,
        momentum,
        weight_decay,
        visualize,
    )

    from mnistnn.data_config import dict_config_test, dict_config_train
    from mnistnn.mnist_neural_network import MNISTNeuralNetwork

    data_loader: MNISTDataLoader = MNISTDataLoader(
        dict_config_train, dict_config_test
    )
    train_images_info: MNISTImagesInfo = data_loader.training_info[0]
    train_labels_info: MNISTLabelsInfo = data_loader.training_info[1]
    test_images_info: MNISTImagesInfo = data_loader.test_info[0]
    test_labels_info: MNISTLabelsInfo = data_loader.test_info[1]
    # The image data is stored as a NumPy array of shape (num_images, rows, cols) 3D
    # The label data is stored as a NumPy array of shape (num_labels,) 1D
    train_images_value: np.ndarray = train_images_info['images']
    train_labels_value: np.ndarray = train_labels_info['labels']
    test_images_value: np.ndarray = test_images_info['images']
    test_labels_value: np.ndarray = test_labels_info['labels']

    if not isinstance(train_images_value, np.ndarray):
        raise TypeError('Training images must be a NumPy array')
    if not isinstance(train_labels_value, np.ndarray):
        raise TypeError('Training labels must be a NumPy array')
    if not isinstance(test_images_value, np.ndarray):
        raise TypeError('Test images must be a NumPy array')
    if not isinstance(test_labels_value, np.ndarray):
        raise TypeError('Test labels must be a NumPy array')

    train_images: np.ndarray = train_images_value
    train_labels: np.ndarray = train_labels_value
    test_images: np.ndarray = test_images_value
    test_labels: np.ndarray = test_labels_value

    prepared_train_images: np.ndarray = _prepare_images(
        train_images, 'Training'
    )
    prepared_test_images: np.ndarray = _prepare_images(test_images, 'Test')
    train_size: int = len(prepared_train_images)
    test_size: int = len(prepared_test_images)
    if train_size == 0 or test_size == 0:
        raise ValueError('Training and test datasets must not be empty')

    if train_labels.ndim != 1 or len(train_labels) != train_size:
        raise ValueError(
            'Training labels must be a 1D array with one label per image'
        )
    if test_labels.ndim != 1 or len(test_labels) != test_size:
        raise ValueError(
            'Test labels must be a 1D array with one label per image'
        )
    if not np.issubdtype(train_labels.dtype, np.integer):
        raise TypeError('Training labels must contain integer class indices')
    if not np.issubdtype(test_labels.dtype, np.integer):
        raise TypeError('Test labels must contain integer class indices')
    if np.any((train_labels < 0) | (train_labels >= 10)):
        raise ValueError(
            'Training labels must be Fashion-MNIST class indices between 0 and 9'
        )
    if np.any((test_labels < 0) | (test_labels >= 10)):
        raise ValueError(
            'Test labels must be Fashion-MNIST class indices between 0 and 9'
        )

    logger.info(
        'Loaded Fashion-MNIST datasets (training: %d images, shape=%s, '
        'dtype=%s; test: %d images, shape=%s, dtype=%s)',
        train_size,
        train_images.shape,
        train_images.dtype,
        test_size,
        test_images.shape,
        test_images.dtype,
    )

    # Visualize the Fashion-MNIST dataset if visualize is True
    visualize_mnist(
        train_images=train_images,
        train_labels=train_labels,
        visualize=visualize,
    )

    network: MNISTNeuralNetwork = MNISTNeuralNetwork()
    rng: np.random.Generator = np.random.default_rng()
    train_images = prepared_train_images
    test_images = prepared_test_images
    # One-hot targets for the 10 Fashion-MNIST classes (labels 0-9)
    # Matrix 10 X 10
    class_targets: np.ndarray = np.eye(10, dtype=np.float64)
    # Train for the specified number of epochs
    training_started_at: float = perf_counter()
    for epoch in range(epochs):
        current_learning_rate: float = _cosine_learning_rate(
            learning_rate, LEARNING_RATE_FLOOR, epoch, epochs
        )
        epoch_started_at: float = perf_counter()
        # Shuffle the training data at the beginning of each epoch
        shuffled_indices: npt.NDArray[np.int64] = rng.permutation(train_size)
        total_loss: float = 0.0

        # Train in batches (the loop is done in batches to avoid memory issues)
        for start in range(0, train_size, batch_size):
            batch_indices: npt.NDArray[np.int64] = shuffled_indices[
                start : start + batch_size
            ]
            batch_images: np.ndarray = _random_horizontal_flip(
                _normalize_images(train_images[batch_indices]),
                rng,
            )
            # Get the correct labels for the current batch
            correct_labels: np.ndarray = train_labels[batch_indices]
            # Get the one-hot encoded target vectors for the correct labels
            batch_targets: np.ndarray = class_targets[correct_labels]
            avg_loss: float = network.train_step(
                batch_images,
                batch_targets,
                current_learning_rate,
                momentum,
                weight_decay,
            )
            total_loss += avg_loss * len(batch_indices)
        # Evaluate the model on the test set after each epoch
        correct_predictions: int = 0
        for start in range(0, test_size, batch_size):
            end: int = start + batch_size
            test_batch_images: np.ndarray = _normalize_images(
                test_images[start:end]
            )
            predictions: np.ndarray = network.predict(
                test_batch_images, tta_hflip=True
            )
            correct_predictions += int(
                np.count_nonzero(predictions == test_labels[start:end])
            )

        average_loss: float = total_loss / train_size
        test_accuracy: float = correct_predictions / test_size
        logger.info(
            'Epoch %d/%d - loss: %.4f - test accuracy: %.2f%% - '
            'learning_rate=%g - duration: %.2fs',
            epoch + 1,
            epochs,
            average_loss,
            test_accuracy * 100,
            current_learning_rate,
            perf_counter() - epoch_started_at,
        )

    logger.info(
        'Fashion-MNIST training completed in %.2fs',
        perf_counter() - training_started_at,
    )
    return network


__all__ = ['train_mnist']
