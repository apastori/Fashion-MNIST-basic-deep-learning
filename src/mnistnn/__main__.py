"""Run Fashion-MNIST visualization and training with ``python -m mnistnn``."""

import logging

from mnistnn.train_mnist import train_mnist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)

train_mnist()
