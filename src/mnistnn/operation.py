"""Define the abstract operation interface for neural-network computations."""

from abc import ABC, abstractmethod

import numpy as np

from mnistnn.utils.shape import assert_same_shape


class Operation(ABC):
    """Base class for operations."""

    def __init__(self) -> None:
        """Initialize the input, output, and gradient state."""
        self.input_: np.ndarray | None = None
        self.output: np.ndarray | None = None
        self.input_grad: np.ndarray | None = None

    def _require_input(self) -> np.ndarray:
        """Return the stored input, or raise if forward has not run."""
        if self.input_ is None:
            raise RuntimeError("Operation input is not available.")
        return self.input_

    def _require_output(self) -> np.ndarray:
        """Return the stored output, or raise if forward has not run."""
        if self.output is None:
            raise RuntimeError("Operation output is not available.")
        return self.output

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Store input in the self._input instance variable.

        Calls the self._output() function.
        """
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Call the self._input_grad() function.

        Checks that the appropriate shapes match.
        """
        ## Check that the output gradient has the same shape as the output
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        ## Check that the input gradient has the same shape as the input
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    @abstractmethod
    def _output(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
