import numpy as np
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
from postprocessing import normalize_output, apply_bandpass_filter, reduce_noise

def load_audio_file(file_path):
    """
    Load an audio file.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        tuple: Loaded audio data and sampling rate.
    """
    data, samplerate = sf.read(file_path)
    return data, samplerate

def save_audio_file(file_path, data, samplerate):
    """
    Save audio data to a file.
    
    Args:
        file_path (str): Path to save the audio file.
        data (np.array): Audio data to save.
        samplerate (int): Sampling rate of the audio data.
    """
    sf.write(file_path, data, samplerate)

def test_tflite_model(tflite_model_path, noisy_data, filter_order):
    """
    Test the TFLite model on noisy data.
    
    Args:
        tflite_model_path (str): Path to the TFLite model file.
        noisy_data (np.array): Noisy input audio data.
        filter_order (int): The order of the adaptive filter.
    
    Returns:
        np.array: Processed output audio data.
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare output array
    num_samples = len(noisy_data)
    output_data = np.zeros(num_samples)

    # Process the input data
    for i in range(filter_order, num_samples):
        input_segment = noisy_data[i-filter_order:i].astype(np.float32).reshape(1, filter_order, 1)
        interpreter.set_tensor(input_details[0]['index'], input_segment)
        interpreter.invoke()
        output_data[i] = interpreter.get_tensor(output_details[0]['index'])

    return output_data

def plot_signals(noisy, predicted, clean, plot_file):
    """
    Plot the noisy, predicted clean, and original clean signals.
    
    Args:
        noisy (np.array): Noisy input signal.
        predicted (np.array): Predicted clean signal from the model.
        clean (np.array): Original clean signal.
        plot_file (str): Path to save the plot image.
    """
    plt.figure(figsize=(15, 8))
    plt.plot(noisy, label='Noisy Signal', alpha=0.6, color='red')
    plt.plot(predicted, label='Predicted Clean Signal', alpha=0.8, color='blue')
    plt.plot(clean, label='Original Clean Signal', alpha=0.8, color='green')
    plt.title('Signal Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.show()

def main():
    # Paths to the noisy input file, clean reference file, output file, and plot file
    noisy_file = r"C:\Users\divya\Desktop\ANC_Project\data\artificial_data\new_test_data\output_sample_15\mixed_output.wav"
    clean_file = r"C:\Users\divya\Desktop\ANC_Project\data\artificial_data\new_test_data\output_sample_15\voice_output.wav"
    output_audio_file = r"C:\Users\divya\Desktop\ANC_Project\data\artificial_data\new_test_data\output_sample_15\filtered_output.wav"
    tflite_model_path = 'anc_model.tflite'
    filter_order = 32
    lowcut = 20.0
    highcut = 7999.0
    fs = 16000  # Sample rate
    plot_output_file = 'signal_plot.png'

    # Load the noisy and clean audio files
    noisy_data, sr = load_audio_file(noisy_file)
    clean_data, _ = load_audio_file(clean_file)

    # Test the TFLite model on the noisy data
    processed_data = test_tflite_model(tflite_model_path, noisy_data, filter_order)

    # Post-processing
    processed_data = apply_bandpass_filter(processed_data, lowcut, highcut, fs)
    processed_data = reduce_noise(processed_data)
    processed_data = normalize_output(processed_data)

    # Save the processed output to a file
    save_audio_file(output_audio_file, processed_data, sr)

    # Plot the signals
    plot_signals(noisy_data, processed_data, clean_data, plot_output_file)

    print(f"Processed output saved to {output_audio_file}")
    print(f"Plot saved to {plot_output_file}")

if __name__ == "__main__":
    main()
