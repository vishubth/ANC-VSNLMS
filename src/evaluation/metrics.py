import numpy as np
from numpy.typing import NDArray


class SignalMetrics:
    """
    DSP evaluation metrics for adaptive ANC benchmarking.
    """

    @staticmethod
    def calculate_snr(
        signal: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> float:
        signal_power = np.mean(np.square(signal))
        noise_power = np.mean(np.square(noise))

        if noise_power == 0:
            return float("inf")

        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def calculate_rmse(
        target_signal: NDArray[np.float64],
        predicted_signal: NDArray[np.float64],
    ) -> float:
        return np.sqrt(
            np.mean(
                np.square(target_signal - predicted_signal)
            )
        )

    @staticmethod
    def calculate_snr_improvement(
        original_snr: float,
        filtered_snr: float,
    ) -> float:
        return filtered_snr - original_snr
