"""Load images and labels from Fashion-MNIST IDX binary data files.

Fashion-MNIST uses the same IDX format as classic MNIST
(magic numbers 2051 for images and 2049 for labels).
"""

import os
from typing import TypedDict

import numpy as np  # linear algebra


class MNISTImagesInfo(TypedDict):
    """Image-file metadata and pixel array loaded from a Fashion-MNIST file."""

    type: str
    num_images: int
    rows: int
    cols: int
    images: np.ndarray


class MNISTLabelsInfo(TypedDict):
    """Label-file metadata and label array loaded from a Fashion-MNIST file."""

    type: str
    num_labels: int
    labels: np.ndarray


MNISTSplitInfo = tuple[MNISTImagesInfo, MNISTLabelsInfo]


#
# Fashion-MNIST Data Loader Class (IDX format compatible with MNIST)
#
class MNISTDataLoader:
    """Class to load Fashion-MNIST dataset from given IDX file paths."""

    def __init__(
        self,
        dict_config_train: dict[str, str],
        dict_config_test: dict[str, str],
    ) -> None:
        """Initialize the loader with training and test file paths."""
        # Initialize file paths from configuration dictionaries
        self.training_input: dict[str, str] = {
            'labels_path': dict_config_train['training_labels_filepath'],
            'images_path': dict_config_train['training_images_filepath'],
        }
        self.test_input: dict[str, str] = {
            'labels_path': dict_config_test['test_labels_filepath'],
            'images_path': dict_config_test['test_images_filepath'],
        }
        # Check if file paths exist
        self._validate_file_paths()
        # Store Training data
        self.training_info: MNISTSplitInfo = self._load_train_data()
        # Store Test data
        self.test_info: MNISTSplitInfo = self._load_test_data()

    """Validate the existence of the Fashion-MNIST data files."""

    # Private method to validate file paths
    # Do not call this method directly from outside the class
    def _validate_file_paths(self) -> None:
        """Validate file existence: training files first, then test files."""
        # Training first
        for name, path in self.training_input.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Training {name.replace('_path', '')} file not found: {path}"
                )

        # Then test
        for name, path in self.test_input.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Test {name.replace('_path', '')} file not found: {path}"
                )

    """Read labels from Fashion-MNIST IDX files."""

    # Private method to validate file paths
    # Do not call this method directly from outside the class
    def _read_labels(self, labels_filepath: str) -> MNISTLabelsInfo:
        labels: list[int] = []
        with open(labels_filepath, "rb") as f:
            # Read and validate magic number header
            magic_number: int = int.from_bytes(f.read(4), "big")
            if magic_number != 2049:
                raise ValueError(
                    f"Invalid label file magic number: {magic_number}"
                )
            # Read number of labels header
            num_labels: int = int.from_bytes(f.read(4), "big")
            # Read label data; each Fashion-MNIST class index 0-9 is one byte
            raw_bytes: bytes = f.read()
            labels = []
            for byte in raw_bytes:
                labels.append(int(byte))
            if any(label < 0 or label > 9 for label in labels):
                raise ValueError(
                    "Labels must be Fashion-MNIST class indices between 0 and 9"
                )
            if len(labels) != num_labels:
                raise ValueError("Label count does not match header")
        labels_dict: MNISTLabelsInfo = {
            "type": "labels",
            "num_labels": num_labels,
            "labels": np.array(labels, dtype=np.uint8),
        }
        return labels_dict

    """Read images from Fashion-MNIST IDX files."""

    # Private method to validate file paths
    # Do not call this method directly from outside the class
    def _read_images(self, images_filepath: str) -> MNISTImagesInfo:
        images: list[np.ndarray] = []
        with open(images_filepath, "rb") as f:
            # Read and validate magic number header
            magic_number: int = int.from_bytes(f.read(4), "big")
            if magic_number != 2051:
                raise ValueError(
                    f"Invalid image file magic number: {magic_number}"
                )
            # Read number of images header
            num_images: int = int.from_bytes(f.read(4), "big")
            # Read number of rows and columns per image
            rows: int = int.from_bytes(f.read(4), "big")
            cols: int = int.from_bytes(f.read(4), "big")
            # Read image data, each pixel is stored as a single byte
            # representing grayscale intensity (0-255)
            image_bytes: bytes = f.read()
            image_bytes_int: list[int] = []
            for byte in image_bytes:
                image_bytes_int.append(int(byte))
            if len(image_bytes_int) != num_images * rows * cols:
                raise ValueError("Image data size does not match header")
            # Build images from flat byte array
            image_size: int = rows * cols
            for i in range(num_images):
                start: int = i * image_size
                end: int = start + image_size
                flat: list[int] = image_bytes_int[start:end]
                if len(flat) != 784:
                    raise ValueError("Invalid image size, expected 784 pixels")
                image: list[list[int]] = []
                for r in range(rows):
                    row: list[int] = flat[r * cols : (r + 1) * cols]
                    image.append(row)
                images.append(np.array(image))
        images_dict: MNISTImagesInfo = {
            "type": "images",
            "num_images": num_images,
            "rows": rows,
            "cols": cols,
            "images": np.array(images, dtype=np.uint8),
        }
        return images_dict

    """Read images and labels from Fashion-MNIST IDX files."""

    # Private method to validate file paths
    # Do not call this method directly from outside the class
    def _read_images_labels(
        self, images_filepath: str, labels_filepath: str
    ) -> MNISTSplitInfo:
        images_dict: MNISTImagesInfo = self._read_images(images_filepath)
        labels_dict: MNISTLabelsInfo = self._read_labels(labels_filepath)
        if images_dict["num_images"] != labels_dict["num_labels"]:
            raise ValueError(
                "Number of images does not match number of labels"
            )
        return (images_dict, labels_dict)

    """Load training data from Fashion-MNIST IDX files."""

    # Private method to validate file paths
    # Do not call this method directly from outside the class
    def _load_train_data(
        self,
    ) -> MNISTSplitInfo:
        return self._read_images_labels(
            self.training_input['images_path'],
            self.training_input['labels_path'],
        )

    """Load test data from Fashion-MNIST IDX files."""

    # Private method to validate file paths
    # Do not call this method directly from outside the class
    def _load_test_data(
        self,
    ) -> MNISTSplitInfo:
        return self._read_images_labels(
            self.test_input['images_path'], self.test_input['labels_path']
        )

    """Load both training and test data from Fashion-MNIST IDX files."""

    def show_data(
        self,
    ) -> tuple[MNISTSplitInfo, MNISTSplitInfo]:
        """Return the loaded training and test data."""
        return (self.training_info, self.test_info)
