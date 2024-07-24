import numpy as np
import sounddevice as sd
import soundfile as sf
from anc_model import ANCModel
import noisereduce as nr
import os
from scipy.signal import butter, lfilter

class RealTimeANC:
    """
    Real-time active noise cancellation class.
    
    Attributes:
        model (ANCModel): The adaptive noise cancellation model.
        sample_rate (int): The sample rate for audio processing.
        filter_order (int): The order of the adaptive filter.
    """
    
    def __init__(self, model, sample_rate=44100, filter_order=32):
        """
        Initialize the RealTimeANC with the given model, sample rate, and filter order.
        
        Args:
            model (ANCModel): The adaptive noise cancellation model.
            sample_rate (int): The sample rate for audio processing.
            filter_order (int): The order of the adaptive filter.
        """
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.model = model

    def record_audio(self, duration, filename='recorded_audio.wav'):
        """
        Record audio for a specified duration and save to a file.
        
        Args:
            duration (int): The duration to record audio in seconds.
            filename (str): The path to the file where the recorded audio will be saved.
        """
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()  # Wait until the recording is finished
        sf.write(filename, recording, self.sample_rate)
        print(f"Recording saved to {filename}")

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_lowpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def process_audio(self, input_filename, output_filename, chunk_size=1024):
        """
        Process the recorded audio with the ANC model and save the processed output.
        
        Args:
            input_filename (str): The path to the input audio file.
            output_filename (str): The path to the output audio file.
            chunk_size (int): The size of the chunks to process the audio in.
        """
        audio, _ = sf.read(input_filename)
        processed_audio = np.zeros_like(audio)

        # Process the audio in larger chunks
        for start_idx in range(0, len(audio) - self.filter_order, chunk_size):
            end_idx = min(start_idx + chunk_size, len(audio))
            chunk = np.array([audio[i-self.filter_order:i][::-1] for i in range(start_idx + self.filter_order, end_idx)])
            
            # Debug: Print the shape and sample values of the chunk
            print(f"Processing chunk from {start_idx} to {end_idx}, chunk shape: {chunk.shape}")
            print(f"Sample input values: {chunk[0]}")

            predictions = self.model.predict(chunk)
            
            # Debug: Print the shape and sample values of the predictions
            print(f"Predictions shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:5]}")

            processed_audio[start_idx + self.filter_order:end_idx] = predictions.flatten()

        # Debug: Check min and max values of processed_audio
        print(f"Processed audio min value: {np.min(processed_audio)}, max value: {np.max(processed_audio)}")

        # Apply bandpass filter to enhance voice and reduce noise
        lowcut = 200.0  # Widened range for more natural sound
        highcut = 4000.0
        processed_audio = self.bandpass_filter(processed_audio, lowcut, highcut, self.sample_rate)

        # Apply noise reduction using spectral gating with less aggressive parameters
        noise_profile = processed_audio[:1000]  # Assume first 1000 samples are noise
        reduced_noise_audio = nr.reduce_noise(y=processed_audio, sr=self.sample_rate, y_noise=noise_profile, prop_decrease=0.8, freq_mask_smooth_hz=200)

        # Apply a low-pass filter to reduce high-frequency static noise
        cutoff = 5000.0
        reduced_noise_audio = self.lowpass_filter(reduced_noise_audio, cutoff, self.sample_rate)

        # Normalize the processed audio
        max_val = np.max(np.abs(reduced_noise_audio))
        if max_val > 0:
            reduced_noise_audio = reduced_noise_audio / max_val

        sf.write(output_filename, reduced_noise_audio, self.sample_rate)
        print(f"Processed audio saved to {output_filename}")

if __name__ == "__main__":
    sample_rate = 44100
    filter_order = 32

    anc_model = ANCModel(input_shape=(filter_order,))
    real_time_anc = RealTimeANC(model=anc_model, sample_rate=sample_rate, filter_order=filter_order)

    # Load pre-trained weights
    weights_path = 'anc_weights.h5'
    anc_model.load_weights(weights_path)

    duration = 10  # Duration of recording in seconds
    recorded_filename = 'recorded_audio.wav'
    output_filename = 'filtered_output.wav'

    # Record and process audio
    real_time_anc.record_audio(duration, recorded_filename)
    real_time_anc.process_audio(recorded_filename, output_filename)
