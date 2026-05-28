import matplotlib.pyplot as plt
import librosa
import librosa.display


class SpectrogramVisualizer:
    """
    Generate spectrogram visualizations for DSP benchmarking.
    """

    @staticmethod
    def generate_spectrogram(audio_path: str) -> None:
        signal, sample_rate = librosa.load(audio_path)

        spectrogram = librosa.amplitude_to_db(
            abs(librosa.stft(signal)),
            ref=max,
        )

        plt.figure(figsize=(10, 5))

        librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            x_axis="time",
            y_axis="log",
        )

        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram Analysis")

        plt.tight_layout()
        plt.show()
