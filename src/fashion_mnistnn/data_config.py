"""Define file paths for the Fashion-MNIST training and test datasets.

The IDX filenames mirror classic MNIST; the files under ``src/data/``
contain Fashion-MNIST images and labels.
"""

import os
from os.path import join

#
# Set file paths based on added Fashion-MNIST datasets
#

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "..", "data")

# Training data file paths dictionary
dict_config_train: dict[str, str] = {
    'input_path': input_path,
    'training_images_filepath': join(
        input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'
    ),
    'training_labels_filepath': join(
        input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    ),
}

# Test data file paths dictionary
dict_config_test: dict[str, str] = {
    'input_path': input_path,
    'test_images_filepath': join(
        input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    ),
    'test_labels_filepath': join(
        input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    ),
}

# Export dictionaries for use in other Python files
__all__: list[str] = ['dict_config_train', 'dict_config_test']
