import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class VSNLMS:
    """
    Variable Step-size Normalized Least Mean Squares (VSNLMS)
    adaptive filtering implementation.

    This module is responsible for:

    - adaptive coefficient optimization
    - dynamic learning-rate adjustment
    - normalized gradient updates
    - convergence stabilization
    - numerical stability safeguards

    The implementation is designed for real-time adaptive
    signal-processing workflows such as Active Noise Cancellation (ANC).
    """

    def __init__(
        self,
        mu: float,
        mu_max: float,
        mu_min: float,
        m0: int,
        m1: int,
        alpha: float,
        delta: float = 0.0,
    ) -> None:
        self.mu = mu
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.m0 = m0
        self.m1 = m1
        self.alpha = alpha
        self.delta = delta
        self.weights: NDArray[np.float64] | None = None

    def adjust_step_size(self, error: float) -> None:
        """
        Dynamically adjust learning rate based on
        current residual error magnitude.

        Args:
            error: Current residual error.
        """

        if abs(error) > self.delta:
            self.mu = min(self.mu * self.alpha, self.mu_max)
        else:
            self.mu = max(self.mu / self.alpha, self.mu_min)

    def update_weights(
        self,
        x: NDArray[np.float64],
        desired_signal: float,
    ) -> Tuple[float, NDArray[np.float64]]:
        """
        Perform a normalized adaptive filter update.

        Args:
            x: Input signal segment.
            desired_signal: Target signal value.

        Returns:
            Tuple containing:
            - residual error
            - updated filter coefficients
        """

        if self.weights is None:
            raise ValueError("Filter weights are not initialized.")

        predicted_output = np.dot(x, self.weights)
        error = desired_signal - predicted_output

        self.adjust_step_size(error)

        signal_energy = np.dot(x, x) + 0.01

        self.weights += (
            self.mu * error * x / signal_energy
        )

        self._stabilize_weights()

        return error, self.weights.copy()

    def _stabilize_weights(self) -> None:
        """
        Prevent coefficient divergence and numerical instability.
        """

        if self.weights is None:
            return

        weight_norm = np.linalg.norm(self.weights)

        if weight_norm > 10:
            self.weights /= weight_norm
