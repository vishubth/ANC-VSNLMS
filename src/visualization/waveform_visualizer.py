import matplotlib.pyplot as plt
import soundfile as sf


class WaveformVisualizer:
    """
    Generate waveform comparison visualizations.
    """

    @staticmethod
    def plot_waveform(audio_path: str) -> None:
        signal, _ = sf.read(audio_path)

        plt.figure(figsize=(12, 4))

        plt.plot(signal)

        plt.title("Waveform Visualization")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")

        plt.grid(True)
        plt.show()
