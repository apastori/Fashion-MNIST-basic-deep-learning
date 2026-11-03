"""Provide inverted dropout for hidden-layer regularization."""

import numpy as np

from mnistnn.operation import Operation
from mnistnn.utils.asarray import as_array


class Dropout(Operation):
    """Drop units during training and scale the remainder by ``1 / (1 - rate)``."""

    def __init__(
        self,
        rate: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize dropout with a drop probability in ``[0, 1)``."""
        if rate < 0.0 or rate >= 1.0:
            raise ValueError('Dropout rate must be in the range [0, 1).')
        super().__init__()
        self.rate: float = rate
        self.training: bool = True
        self.mask: np.ndarray | None = None
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    def _output(self) -> np.ndarray:
        """Apply inverted dropout in train mode; identity in eval mode."""
        input_: np.ndarray = self._require_input()
        if (not self.training) or self.rate == 0.0:
            self.mask = None
            return input_

        keep_prob: float = 1.0 - self.rate
        # Keep Prob is a boolean np.array of the same shape as input_
        keep_mask: np.ndarray = self.rng.random(input_.shape) < keep_prob
        # Mask is a numerical np.array of the same shape as input_
        # that is like [1.0, 0.0, 1.0, 1.0, 1.0]
        self.mask = keep_mask.astype(input_.dtype, copy=False) / keep_prob
        return as_array(input_ * self.mask)

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Route gradients through the dropout mask from the forward pass."""
        if self.mask is None:
            return as_array(grad)
        return as_array(grad * self.mask)
