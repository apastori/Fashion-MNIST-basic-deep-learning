"""Provide cross-entropy loss for multi-class classification."""

import numpy as np

from mnistnn.loss import Loss
from mnistnn.utils.asarray import as_array


class CrossEntropyLoss(Loss):
    """Calculate cross-entropy loss for multi-class classification."""

    def _output(self) -> float:
        """Calculate loss from softmax predictions and one-hot targets.

        Predictions have shape ``(batch_size, num_classes)``. Targets are
        one-hot encoded labels with the same shape.
        """
        if self.prediction is None or self.target is None:
            raise RuntimeError('Loss inputs are not available.')

        # Clip predictions to avoid log(0)
        clipped_preds: np.ndarray = np.clip(self.prediction, 1e-10, 1 - 1e-10)

        # Compute cross-entropy loss
        loss = (
            np.sum(self.target * np.log(clipped_preds)) / self.target.shape[0]
        ) * -1
        return float(loss)

    def _input_grad(self) -> np.ndarray:
        """Compute the gradient of cross-entropy with respect to probabilities."""
        if self.prediction is None or self.target is None:
            raise RuntimeError('Loss inputs are not available.')
        clipped_preds: np.ndarray = np.clip(self.prediction, 1e-10, 1 - 1e-10)
        return as_array(
            (self.target / clipped_preds / self.target.shape[0]) * -1
        )
