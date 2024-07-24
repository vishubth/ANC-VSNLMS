import numpy as np
import sounddevice as sd
import soundfile as sf
from adaptive_filter import AdaptiveFilter
import os

class LiveAudioFilter:
    def __init__(self, filter_order, vsnlms_params, weights_path=None, sample_rate=44100):
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.adaptive_filter = AdaptiveFilter(filter_order, vsnlms_params, weights_path)
        self.output_data = []

    def record_audio(self, duration, temp_filename='temp_recording.wav'):
        """
        Record audio from the microphone and save it to a temporary file.

        Args:
            duration (int): Duration of the recording in seconds.
            temp_filename (str): Path to the temporary file to save the recording.
        """
        try:
            print(f"Recording for {duration} seconds...")
            recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
            sd.wait()
            sf.write(temp_filename, recording, self.sample_rate)
            print(f"Recording saved to {temp_filename}")
        except Exception as e:
            print(f"An error occurred during recording: {e}")

    def process_audio(self, input_filename, output_filename):
        """
        Process the recorded audio using the adaptive filter and save the output.

        Args:
            input_filename (str): Path to the input file (recorded audio).
            output_filename (str): Path to the output file (filtered audio).
        """
        try:
            noisy_signal, _ = sf.read(input_filename)
            noisy_signal = noisy_signal.flatten()  # Flatten in case it's not a 1D array

            # Preprocess the noisy signal (optional)
            max_val = np.max(np.abs(noisy_signal))
            if max_val > 0:
                noisy_signal = noisy_signal / max_val

            print(f"Max value of noisy signal: {max_val}")
            print(f"First 10 samples of noisy signal: {noisy_signal[:10]}")

            # Ensure the input length matches the filter order
            if len(noisy_signal) < self.filter_order:
                print(f"Input signal length ({len(noisy_signal)}) is less than filter order ({self.filter_order}). Padding the signal.")
                padded_noisy_signal = np.pad(noisy_signal, (self.filter_order, 0), 'constant', constant_values=(0, 0))
            else:
                padded_noisy_signal = noisy_signal

            filtered_output = np.zeros_like(padded_noisy_signal)
            for n in range(self.filter_order, len(padded_noisy_signal)):
                x = padded_noisy_signal[n-self.filter_order:n][::-1]
                if x.shape[0] != self.adaptive_filter.vsnlms.weights.shape[0]:
                    print(f"Shape mismatch: segment shape {x.shape}, weights shape {self.adaptive_filter.vsnlms.weights.shape}")
                filtered_output[n] = np.dot(self.adaptive_filter.vsnlms.weights, x)

            filtered_output = filtered_output[:len(noisy_signal)]  # Remove any extra padding if it was added

            print(f"First 10 samples of filtered output: {filtered_output[:10]}")
            
            sf.write(output_filename, filtered_output, self.sample_rate)
            print(f"Filtered audio saved to {output_filename}")
        except Exception as e:
            print(f"An error occurred during processing: {e}")

    def start(self, duration=10, temp_filename='temp_recording.wav', output_filename='filtered_output.wav'):
        """
        Record, process, and save the filtered audio.

        Args:
            duration (int): Duration of the recording in seconds.
            temp_filename (str): Path to the temporary file to save the recording.
            output_filename (str): Path to the output file to save the filtered audio.
        """
        self.record_audio(duration, temp_filename)
        self.process_audio(temp_filename, output_filename)

        # Optionally, delete the temporary file after processing
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"Temporary file {temp_filename} deleted.")

if __name__ == "__main__":
    filter_order = 64  # Experiment with different filter orders
    vsnlms_params = (0.0001, 0.1, 0.000001, 10, 10, 1.01)  # Adjusted parameters
    weights_path = 'adaptive_filter_weights.npy'
    live_audio_filter = LiveAudioFilter(filter_order, vsnlms_params, weights_path)

    duration = 10  # Duration of recording in seconds
    temp_filename = 'temp_recording.wav'
    output_filename = 'filtered_output.wav'
    live_audio_filter.start(duration, temp_filename, output_filename)
