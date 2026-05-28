import logging
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.typing import NDArray

from src.filters.adaptive_filter import AdaptiveFilter


logger = logging.getLogger(__name__)


class LiveAudioFilter:
    """
    Real-time ANC runtime processing pipeline.

    Responsibilities:

    - microphone audio capture
    - runtime preprocessing
    - adaptive filtering execution
    - runtime DSP orchestration
    - filtered output generation
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

    def record_audio(
        self,
        duration: int,
        temp_filename: str = "temp_recording.wav",
    ) -> None:
        try:
            logger.info(
                "Starting realtime audio capture for %s seconds.",
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

        except Exception as runtime_error:
            logger.error(
                "Realtime audio capture failed: %s",
                runtime_error,
            )
            raise

    def process_audio(
        self,
        input_filename: str,
        output_filename: str,
    ) -> None:
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
                "Filtered audio saved to %s",
                output_filename,
            )

        except Exception as runtime_error:
            logger.error(
                "Realtime DSP pipeline failed: %s",
                runtime_error,
            )
            raise

    def start(
        self,
        duration: int = 10,
        temp_filename: str = "temp_recording.wav",
        output_filename: str = "filtered_output.wav",
    ) -> None:
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
        max_amplitude = np.max(np.abs(signal))

        if max_amplitude > 0:
            signal = signal / max_amplitude

        return signal

    def _prepare_signal(
        self,
        signal: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if len(signal) < self.filter_order:
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
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

            logger.info(
                "Temporary runtime artifact removed: %s",
                temp_filename,
            )
