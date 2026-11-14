"""Run Fashion-MNIST visualization and training with ``python -m fashion_mnistnn``."""

import logging

from fashion_mnistnn.train_fashion_mnist import train_fashion_mnist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)

train_fashion_mnist()
