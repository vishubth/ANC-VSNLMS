import logging
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from vsnlms import VSNLMS


logger = logging.getLogger(__name__)


class AdaptiveFilter:
    """
    Adaptive filtering orchestration layer built on top of
    the VSNLMS optimization algorithm.

    Responsibilities:

    - adaptive training orchestration
    - runtime filter execution
    - convergence tracking
    - filter state persistence
    - DSP experimentation support

    This module acts as the primary runtime abstraction layer
    for adaptive signal-processing workflows.
    """

    def __init__(
        self,
        filter_order: int,
        vsnlms_params: tuple,
        weights_path: str | None = None,
    ) -> None:
        self.filter_order = filter_order
        self.vsnlms = VSNLMS(*vsnlms_params)

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            self.vsnlms.weights = np.zeros(filter_order)

        self.total_error: float = 0.0
        self.num_samples: int = 0
        self.rmse_history: List[float] = []
        self.weights_path = weights_path

    def train(
        self,
        noisy_signal: NDArray[np.float64],
        clean_signal: NDArray[np.float64],
    ) -> None:
        """
        Train adaptive coefficients using noisy and target signals.

        Args:
            noisy_signal: Noisy input signal.
            clean_signal: Desired target signal.
        """

        try:
            total_samples = min(len(noisy_signal), len(clean_signal))

            batch_id = random.randint(1000, 99999)

            self.total_error = 0.0
            self.num_samples = 0

            for index in range(self.filter_order, total_samples):
                signal_window = noisy_signal[
                    index - self.filter_order:index
                ]

                if len(signal_window.shape) > 1:
                    signal_window = signal_window.ravel()

                desired_signal = clean_signal[index]

                error, _ = self.vsnlms.update_weights(
                    signal_window,
                    desired_signal,
                )

                self.total_error += error ** 2
                self.num_samples += 1

            if self.weights_path:
                self.save_weights(self.weights_path)

            self._record_training_metrics(batch_id)

        except Exception as runtime_error:
            logger.error(
                "Adaptive filter training failed: %s",
                runtime_error,
            )
            raise

    def apply_filter(
        self,
        input_signal: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Apply learned adaptive coefficients to an input signal.

        Args:
            input_signal: Signal to be filtered.

        Returns:
            Filtered output signal.
        """

        try:
            if self.weights_path and os.path.exists(self.weights_path):
                self.load_weights(self.weights_path)

            total_samples = len(input_signal)
            output_signal = np.zeros(total_samples)

            for index in range(self.filter_order, total_samples):
                signal_window = input_signal[
                    index - self.filter_order:index
                ][::-1]

                output_signal[index] = np.dot(
                    self.vsnlms.weights,
                    signal_window,
                )

            return output_signal

        except Exception as runtime_error:
            logger.error(
                "Adaptive filtering failed: %s",
                runtime_error,
            )
            raise

    def save_weights(self, filename: str) -> None:
        """
        Persist adaptive coefficients to disk.
        """

        try:
            np.save(filename, self.vsnlms.weights)

        except Exception as runtime_error:
            logger.error(
                "Failed to save weights to %s: %s",
                filename,
                runtime_error,
            )
            raise

    def load_weights(self, filename: str) -> None:
        """
        Restore adaptive coefficients from disk.
        """

        try:
            self.vsnlms.weights = np.load(filename)

        except Exception as runtime_error:
            logger.error(
                "Failed to load weights from %s: %s",
                filename,
                runtime_error,
            )
            raise

    def plot_performance(self) -> None:
        """
        Visualize RMSE convergence over training iterations.
        """

        try:
            if not self.rmse_history:
                logger.warning("No RMSE history available for plotting.")
                return

            plt.figure(figsize=(10, 5))
            plt.plot(self.rmse_history)
            plt.title("RMSE Convergence Over Training Batches")
            plt.xlabel("Batch Number")
            plt.ylabel("Root Mean Square Error")
            plt.grid(True)
            plt.show()

        except Exception as runtime_error:
            logger.error(
                "Failed to generate convergence plot: %s",
                runtime_error,
            )
            raise

    def _record_training_metrics(self, batch_id: int) -> None:
        """
        Compute and log training convergence statistics.
        """

        average_error = (
            (self.total_error / self.num_samples) ** 0.5
            if self.num_samples > 0
            else float("inf")
        )

        self.rmse_history.append(average_error)

        logger.info("=" * 60)
        logger.info(
            "Training batch %s completed.",
            batch_id,
        )
        logger.info(
            "Processed samples: %s",
            self.num_samples,
        )
        logger.info(
            "RMSE convergence metric: %.6f",
            average_error,
        )
