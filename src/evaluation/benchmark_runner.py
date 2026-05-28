import logging

from src.evaluation.metrics import SignalMetrics


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Benchmark execution pipeline for ANC evaluation.
    """

    @staticmethod
    def evaluate(
        clean_signal,
        noisy_signal,
        filtered_signal,
    ) -> dict:
        original_noise = noisy_signal - clean_signal
        filtered_noise = filtered_signal - clean_signal

        original_snr = SignalMetrics.calculate_snr(
            clean_signal,
            original_noise,
        )

        filtered_snr = SignalMetrics.calculate_snr(
            clean_signal,
            filtered_noise,
        )

        snr_improvement = (
            SignalMetrics.calculate_snr_improvement(
                original_snr,
                filtered_snr,
            )
        )

        rmse = SignalMetrics.calculate_rmse(
            clean_signal,
            filtered_signal,
        )

        benchmark_results = {
            "original_snr": original_snr,
            "filtered_snr": filtered_snr,
            "snr_improvement": snr_improvement,
            "rmse": rmse,
        }

        logger.info(
            "Benchmark evaluation completed: %s",
            benchmark_results,
        )

        return benchmark_results
