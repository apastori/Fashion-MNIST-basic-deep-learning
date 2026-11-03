"""Tests for Fashion-MNIST data loader class."""

import numpy as np
import pytest

from mnistnn.data_config import dict_config_test, dict_config_train
from mnistnn.mnist_data_loader import (
    MNISTDataLoader,
    MNISTImagesInfo,
    MNISTLabelsInfo,
    MNISTSplitInfo,
)


class TestMNISTDataLoader:
    """Test suite for Fashion-MNIST data loader class."""

    @pytest.fixture
    def data_loader(self) -> MNISTDataLoader:
        """Initialize Fashion-MNIST data loader with configuration."""
        loader: MNISTDataLoader = MNISTDataLoader(
            dict_config_train, dict_config_test
        )
        return loader

    def test_loader_initialization(self, data_loader: MNISTDataLoader) -> None:
        """Test that data loader initializes without errors."""
        if data_loader is None:
            raise AssertionError("data_loader is None")
        if data_loader.training_info is None:
            raise AssertionError("data_loader.training_info is None")
        if data_loader.test_info is None:
            raise AssertionError("data_loader.test_info is None")

    # ===== Training Data Tests =====

    def test_training_data_structure(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that training data has correct structure (images_dict, labels_dict)."""
        training_images_dict: MNISTImagesInfo
        training_labels_dict: MNISTLabelsInfo
        training_images_dict, training_labels_dict = data_loader.training_info

        if not isinstance(training_images_dict, dict):
            raise AssertionError(
                f"Expected training_images_dict to be dict, "
                f"got {type(training_images_dict).__name__}"
            )

        if not isinstance(training_labels_dict, dict):
            raise AssertionError(
                f"Expected training_labels_dict to be dict, "
                f"got {type(training_labels_dict).__name__}"
            )

    def test_training_images_magic_number(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that training images file has correct magic number (2051)."""
        # Magic number validation happens during loading
        # If loading succeeds, the magic number was valid (2051)
        training_images_dict: MNISTImagesInfo
        training_images_dict, _ = data_loader.training_info
        if training_images_dict['type'] != 'images':
            raise AssertionError(
                f"Expected training_images_dict type to be 'images', "
                f"got {training_images_dict['type']}"
            )

    def test_training_labels_magic_number(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that training labels file has correct magic number (2049)."""
        # Magic number validation happens during loading
        # If loading succeeds, the magic number was valid (2049)
        training_labels_dict: MNISTLabelsInfo
        _, training_labels_dict = data_loader.training_info
        if training_labels_dict['type'] != 'labels':
            raise AssertionError(
                f"Expected training_labels_dict type to be 'labels', "
                f"got {training_labels_dict['type']}"
            )

    def test_training_images_count_matches_header(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that number of training images matches header."""
        training_images_dict: MNISTImagesInfo
        training_images_dict, _ = data_loader.training_info
        num_images: int = training_images_dict['num_images']
        images_array: np.ndarray = training_images_dict['images']
        if len(images_array) != num_images:
            raise AssertionError(
                f"Expected {num_images} images, got {len(images_array)}"
            )
        if images_array.shape[0] != num_images:
            raise AssertionError(
                f"Expected array shape[0] to be {num_images}, "
                f"got {images_array.shape[0]}"
            )

    def test_training_labels_count_matches_header(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that number of training labels matches header."""
        training_labels_dict: MNISTLabelsInfo
        _, training_labels_dict = data_loader.training_info
        num_labels: int = training_labels_dict['num_labels']
        labels_array: np.ndarray = training_labels_dict['labels']
        if len(labels_array) != num_labels:
            raise AssertionError(
                f"Expected {num_labels} labels, got {len(labels_array)}"
            )
        if labels_array.shape[0] != num_labels:
            raise AssertionError(
                f"Expected array shape[0] to be {num_labels}, "
                f"got {labels_array.shape[0]}"
            )

    def test_training_images_and_labels_count_match(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that number of training images equals number of labels."""
        training_images_dict: MNISTImagesInfo
        training_labels_dict: MNISTLabelsInfo
        training_images_dict, training_labels_dict = data_loader.training_info
        num_images: int = training_images_dict['num_images']
        num_labels: int = training_labels_dict['num_labels']
        if num_images != num_labels:
            raise AssertionError(
                f"Expected {num_images} images and {num_images} labels, "
                f"but got {num_labels} labels"
            )

    def test_training_image_dimensions(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that training images have correct dimensions (28x28)."""
        training_images_dict: MNISTImagesInfo
        training_images_dict, _ = data_loader.training_info
        rows: int = training_images_dict['rows']
        cols: int = training_images_dict['cols']
        if rows != 28:
            raise AssertionError(
                f"Expected training image rows to be 28, got {rows}"
            )
        if cols != 28:
            raise AssertionError(
                f"Expected training image cols to be 28, got {cols}"
            )

    def test_training_image_shape(self, data_loader: MNISTDataLoader) -> None:
        """Test that each training image is 28x28."""
        training_images_dict: MNISTImagesInfo
        training_images_dict, _ = data_loader.training_info
        images_array: np.ndarray = training_images_dict['images']
        for idx, image in enumerate(images_array):
            image_element = image  # type: np.ndarray
            if image_element.shape != (28, 28):
                raise AssertionError(
                    f"Expected training image {idx} shape to be (28, 28), "
                    f"got {image.shape}"
                )

    def test_training_labels_are_digits(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that training labels are Fashion-MNIST class indices (0-9)."""
        training_labels_dict: MNISTLabelsInfo
        _, training_labels_dict = data_loader.training_info
        labels_array: np.ndarray = training_labels_dict['labels']
        if not np.all(labels_array >= 0):
            raise AssertionError(
                f"Expected all training labels >= 0, "
                f"found min: {np.min(labels_array)}"
            )
        if not np.all(labels_array <= 9):
            raise AssertionError(
                f"Expected all training labels <= 9, "
                f"found max: {np.max(labels_array)}"
            )

    def test_training_image_pixel_values(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that training image pixels are valid grayscale values (0-255)."""
        training_images_dict: MNISTImagesInfo
        training_images_dict, _ = data_loader.training_info
        images_array: np.ndarray = training_images_dict['images']
        if not np.all(images_array >= 0):
            raise AssertionError(
                f"Expected all training image pixels >= 0, "
                f"found min: {np.min(images_array)}"
            )
        if not np.all(images_array <= 255):
            raise AssertionError(
                f"Expected all training image pixels <= 255, "
                f"found max: {np.max(images_array)}"
            )

    # ===== Test Data Tests =====

    def test_test_data_structure(self, data_loader: MNISTDataLoader) -> None:
        """Test that test data has correct structure (images_dict, labels_dict)."""
        test_images_dict: MNISTImagesInfo
        test_labels_dict: MNISTLabelsInfo
        test_images_dict, test_labels_dict = data_loader.test_info
        if not isinstance(test_images_dict, dict):
            raise AssertionError(
                f"Expected test_images_dict to be dict, "
                f"got {type(test_images_dict).__name__}"
            )
        if not isinstance(test_labels_dict, dict):
            raise AssertionError(
                f"Expected test_labels_dict to be dict, "
                f"got {type(test_labels_dict).__name__}"
            )

    def test_test_images_magic_number(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that test images file has correct magic number (2051)."""
        # Magic number validation happens during loading
        # If loading succeeds, the magic number was valid (2051)
        test_images_dict: MNISTImagesInfo
        test_images_dict, _ = data_loader.test_info
        if test_images_dict['type'] != 'images':
            raise AssertionError(
                f"Expected test_images_dict type to be 'images', "
                f"got {test_images_dict['type']}"
            )

    def test_test_labels_magic_number(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that test labels file has correct magic number (2049)."""
        # Magic number validation happens during loading
        # If loading succeeds, the magic number was valid (2049)
        test_labels_dict: MNISTLabelsInfo
        _, test_labels_dict = data_loader.test_info
        if test_labels_dict['type'] != 'labels':
            raise AssertionError(
                f"Expected test_labels_dict type to be 'labels', "
                f"got {test_labels_dict['type']}"
            )

    def test_test_images_count_matches_header(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that number of test images matches header."""
        test_images_dict: MNISTImagesInfo
        test_images_dict, _ = data_loader.test_info
        num_images: int = test_images_dict['num_images']
        images_array: np.ndarray = test_images_dict['images']
        if len(images_array) != num_images:
            raise AssertionError(
                f"Expected {num_images} test images, got {len(images_array)}"
            )
        if images_array.shape[0] != num_images:
            raise AssertionError(
                f"Expected test array shape[0] to be {num_images}, "
                f"got {images_array.shape[0]}"
            )

    def test_test_labels_count_matches_header(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that number of test labels matches header."""
        test_labels_dict: MNISTLabelsInfo
        _, test_labels_dict = data_loader.test_info
        num_labels: int = test_labels_dict['num_labels']
        labels_array: np.ndarray = test_labels_dict['labels']
        if len(labels_array) != num_labels:
            raise AssertionError(
                f"Expected {num_labels} test labels, got {len(labels_array)}"
            )
        if labels_array.shape[0] != num_labels:
            raise AssertionError(
                f"Expected test array shape[0] to be {num_labels}, "
                f"got {labels_array.shape[0]}"
            )

    def test_test_images_and_labels_count_match(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that number of test images equals number of test labels."""
        test_images_dict: MNISTImagesInfo
        test_labels_dict: MNISTLabelsInfo
        test_images_dict, test_labels_dict = data_loader.test_info
        num_images: int = test_images_dict['num_images']
        num_labels: int = test_labels_dict['num_labels']
        if num_images != num_labels:
            raise AssertionError(
                f"Expected {num_images} test images and {num_images} labels, "
                f"but got {num_labels} labels"
            )

    def test_test_image_dimensions(self, data_loader: MNISTDataLoader) -> None:
        """Test that test images have correct dimensions (28x28)."""
        test_images_dict: MNISTImagesInfo
        test_images_dict, _ = data_loader.test_info
        rows: int = test_images_dict['rows']
        cols: int = test_images_dict['cols']
        if rows != 28:
            raise AssertionError(
                f"Expected test image rows to be 28, got {rows}"
            )
        if cols != 28:
            raise AssertionError(
                f"Expected test image cols to be 28, got {cols}"
            )

    def test_test_image_shape(self, data_loader: MNISTDataLoader) -> None:
        """Test that each test image is 28x28."""
        test_images_dict: MNISTImagesInfo
        test_images_dict, _ = data_loader.test_info
        images_array: np.ndarray = test_images_dict['images']
        for idx, image in enumerate(images_array):
            if image.shape != (28, 28):
                raise AssertionError(
                    f"Expected test image {idx} shape to be (28, 28), "
                    f"got {image.shape}"
                )

    def test_test_labels_are_digits(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that test labels are Fashion-MNIST class indices (0-9)."""
        test_labels_dict: MNISTLabelsInfo
        _, test_labels_dict = data_loader.test_info
        labels_array: np.ndarray = test_labels_dict['labels']
        if not np.all(labels_array >= 0):
            raise AssertionError(
                f"Expected all test labels >= 0, "
                f"found min: {np.min(labels_array)}"
            )
        if not np.all(labels_array <= 9):
            raise AssertionError(
                f"Expected all test labels <= 9, "
                f"found max: {np.max(labels_array)}"
            )

    def test_test_image_pixel_values(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that test image pixels are valid grayscale values (0-255)."""
        test_images_dict: MNISTImagesInfo
        test_images_dict, _ = data_loader.test_info
        images_array: np.ndarray = test_images_dict['images']
        if not np.all(images_array >= 0):
            raise AssertionError(
                f"Expected all test image pixels >= 0, "
                f"found min: {np.min(images_array)}"
            )
        if not np.all(images_array <= 255):
            raise AssertionError(
                f"Expected all test image pixels <= 255, "
                f"found max: {np.max(images_array)}"
            )

    # ===== Show Data Method Tests =====

    def test_show_data_returns_both_datasets(
        self, data_loader: MNISTDataLoader
    ) -> None:
        """Test that show_data returns both training and test data."""
        training_info: MNISTSplitInfo
        test_info: MNISTSplitInfo
        training_info, test_info = data_loader.show_data()
        if training_info is None:
            raise AssertionError("Expected training_info to not be None")
        if test_info is None:
            raise AssertionError("Expected test_info to not be None")
        if not isinstance(training_info, tuple):
            raise AssertionError(
                f"Expected training_info to be tuple, "
                f"got {type(training_info).__name__}"
            )
        if not isinstance(test_info, tuple):
            raise AssertionError(
                f"Expected test_info to be tuple, "
                f"got {type(test_info).__name__}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
