import logging
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.typing import NDArray

from adaptive_filter import AdaptiveFilter


logger = logging.getLogger(__name__)


class LiveAudioFilter:
    """
    Real-time audio runtime pipeline for adaptive
    Active Noise Cancellation (ANC).

    Responsibilities:

    - microphone audio acquisition
    - runtime signal preprocessing
    - adaptive filter execution
    - filtered audio generation
    - streaming-oriented DSP experimentation

    This module simulates deployment-style ANC runtime behavior.
    """

    def __init__(
        self,
        filter_order: int,
        vsnlms_params: tuple,
        weights_path: str | None = None,
        sample_rate: int = 44100,
    ) -> None:
        self.sample_rate = sample_rate
        self.filter_order = filter_order

        self.adaptive_filter = AdaptiveFilter(
            filter_order,
            vsnlms_params,
            weights_path,
        )

        self.output_data: list = []

    def record_audio(
        self,
        duration: int,
        temp_filename: str = "temp_recording.wav",
    ) -> None:
        """
        Capture microphone audio and persist temporary runtime input.

        Args:
            duration: Recording duration in seconds.
            temp_filename: Temporary recording path.
        """

        try:
            logger.info(
                "Starting live audio capture for %s seconds.",
                duration,
            )

            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )

            sd.wait()

            sf.write(
                temp_filename,
                recording,
                self.sample_rate,
            )

            logger.info(
                "Audio recording persisted to %s",
                temp_filename,
            )

        except Exception as runtime_error:
            logger.error(
                "Live audio capture failed: %s",
                runtime_error,
            )
            raise

    def process_audio(
        self,
        input_filename: str,
        output_filename: str,
    ) -> None:
        """
        Execute runtime adaptive filtering pipeline.

        Args:
            input_filename: Input audio path.
            output_filename: Output filtered audio path.
        """

        try:
            noisy_signal, _ = sf.read(input_filename)

            noisy_signal = noisy_signal.flatten()

            normalized_signal = self._normalize_signal(
                noisy_signal,
            )

            padded_signal = self._prepare_signal(
                normalized_signal,
            )

            filtered_output = self._execute_filtering_pipeline(
                padded_signal,
            )

            filtered_output = filtered_output[
                : len(normalized_signal)
            ]

            sf.write(
                output_filename,
                filtered_output,
                self.sample_rate,
            )

            logger.info(
                "Filtered runtime audio saved to %s",
                output_filename,
            )

        except Exception as runtime_error:
            logger.error(
                "Runtime audio processing failed: %s",
                runtime_error,
            )
            raise

    def start(
        self,
        duration: int = 10,
        temp_filename: str = "temp_recording.wav",
        output_filename: str = "filtered_output.wav",
    ) -> None:
        """
        Execute end-to-end live ANC runtime workflow.
        """

        self.record_audio(duration, temp_filename)

        self.process_audio(
            temp_filename,
            output_filename,
        )

        self._cleanup_temp_file(temp_filename)

    def _normalize_signal(
        self,
        signal: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Normalize runtime signal amplitude.
        """

        max_amplitude = np.max(np.abs(signal))

        if max_amplitude > 0:
            signal = signal / max_amplitude

        logger.info(
            "Signal normalization completed. Max amplitude: %.6f",
            max_amplitude,
        )

        return signal

    def _prepare_signal(
        self,
        signal: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Prepare runtime signal for adaptive filtering.
        """

        if len(signal) < self.filter_order:
            logger.warning(
                "Signal shorter than filter order. Applying zero-padding."
            )

            return np.pad(
                signal,
                (self.filter_order, 0),
                "constant",
                constant_values=(0, 0),
            )

        return signal

    def _execute_filtering_pipeline(
        self,
        padded_signal: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Execute incremental adaptive filtering pipeline.
        """

        filtered_output = np.zeros_like(padded_signal)

        for index in range(
            self.filter_order,
            len(padded_signal),
        ):
            signal_window = padded_signal[
                index - self.filter_order:index
            ][::-1]

            filtered_output[index] = np.dot(
                self.adaptive_filter.vsnlms.weights,
                signal_window,
            )

        return filtered_output

    def _cleanup_temp_file(self, temp_filename: str) -> None:
        """
        Remove temporary runtime artifacts.
        """

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

            logger.info(
                "Temporary runtime artifact removed: %s",
                temp_filename,
            )


if __name__ == "__main__":
    FILTER_ORDER = 64

    VSNLMS_PARAMS = (
        0.0001,
        0.1,
        0.000001,
        10,
        10,
        1.01,
    )

    WEIGHTS_PATH = "adaptive_filter_weights.npy"

    runtime_filter = LiveAudioFilter(
        FILTER_ORDER,
        VSNLMS_PARAMS,
        WEIGHTS_PATH,
    )

    runtime_filter.start(
        duration=10,
        temp_filename="temp_recording.wav",
        output_filename="filtered_output.wav",
    )
