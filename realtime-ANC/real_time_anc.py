import numpy as np
import sounddevice as sd
import soundfile as sf
from anc_model import ANCModel
import os

class RealTimeANC:
    """
    Real-time active noise cancellation class.
    
    Attributes:
        model (ANCModel): The adaptive noise cancellation model.
        sample_rate (int): The sample rate for audio processing.
        chunk_size (int): The size of audio chunks to process.
        filter_order (int): The order of the adaptive filter.
        output_data (list): The list to store processed audio data.
    """
    
    def __init__(self, model, sample_rate=44100, chunk_size=1024, filter_order=32):
        """
        Initialize the RealTimeANC with the given model, sample rate, chunk size, and filter order.
        
        Args:
            model (ANCModel): The adaptive noise cancellation model.
            sample_rate (int): The sample rate for audio processing.
            chunk_size (int): The size of audio chunks to process.
            filter_order (int): The order of the adaptive filter.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.filter_order = filter_order
        self.model = model
        self.output_data = []

    def process_chunk(self, indata, outdata, frames, time, status):
        """
        Process each audio chunk in real-time.
        
        Args:
            indata (np.array): The input audio data.
            outdata (np.array): The output audio data.
            frames (int): The number of frames.
            time (CData): The time of the audio chunk.
            status (CallbackFlags): The status of the audio stream.
        """
        if status:
            print(f"Stream status: {status}")

        if indata.shape[0] == 0:
            print("No input data")
            return

        chunk = indata[:, 0]
        processed_chunk = np.zeros_like(chunk)

        # Process the chunk in batches
        for i in range(self.filter_order, len(chunk), self.chunk_size):
            end_index = min(i + self.chunk_size, len(chunk))
            batch = np.array([chunk[j-self.filter_order:j] for j in range(i, end_index)]).reshape(-1, self.filter_order)
            predictions = self.model.predict(batch)
            processed_chunk[i:end_index] = predictions.flatten()

        outdata[:, 0] = processed_chunk
        self.output_data.extend(processed_chunk.tolist())

    def start(self, duration=10, output_filename='filtered_output.wav'):
        """
        Start the real-time processing of audio.
        
        Args:
            duration (int): The duration of processing in seconds.
            output_filename (str): The path to the file where the filtered audio will be saved.
        """
        try:
            with sd.Stream(callback=self.process_chunk, channels=1, samplerate=self.sample_rate, blocksize=self.chunk_size):
                print(f"Processing for {duration} seconds...")
                sd.sleep(int(duration * 1000))
            self.save_output(output_filename)
        except Exception as e:
            print(f"An error occurred: {e}")

    def save_output(self, filename):
        """
        Save the processed audio to a file.
        
        Args:
            filename (str): The path to the file where the processed audio will be saved.
        """
        if not self.output_data:
            print("No data to save")
            return

        sf.write(filename, np.array(self.output_data), self.sample_rate)
        print(f"Filtered audio saved to {filename}")

if __name__ == "__main__":
    sample_rate = 44100
    chunk_size = 1024
    filter_order = 32

    anc_model = ANCModel(input_shape=(filter_order,))
    real_time_anc = RealTimeANC(model=anc_model, sample_rate=sample_rate, chunk_size=chunk_size, filter_order=filter_order)

    # Load pre-trained weights
    weights_path = 'anc_weights.h5'
    anc_model.load_weights(weights_path)

    duration = 10  # Duration of processing in seconds
    output_filename = 'filtered_output.wav'

    real_time_anc.start(duration, output_filename)
