import numpy as np
import soundfile as sf
from anc_model import ANCModel
import os

def find_min_length(directory):
    """
    Find the minimum length of audio files in the directory structure.
    
    Args:
        directory (str): Path to the root directory containing subdirectories with noisy and clean audio files.
        
    Returns:
        int: The minimum length of the audio files.
    """
    min_length = float('inf')

    for subdir, _, files in os.walk(directory):
        for file in files:
            if 'mixed_output.wav' in file or 'voice_output.wav' in file:
                file_path = os.path.join(subdir, file)
                data, _ = sf.read(file_path)
                if len(data) < min_length:
                    min_length = len(data)

    return min_length

def load_and_preprocess_audio(subdir, target_length):
    """
    Load and preprocess audio files from a subdirectory.
    
    Args:
        subdir (str): Path to the subdirectory containing noisy and clean audio files.
        target_length (int): The target length for all audio files.
        
    Returns:
        tuple: Lists of noisy and clean audio data.
    """
    noisy_file = None
    clean_file = None
    for file in os.listdir(subdir):
        if 'mixed_output.wav' in file:
            noisy_file = os.path.join(subdir, file)
        elif 'voice_output.wav' in file:
            clean_file = os.path.join(subdir, file)

    if noisy_file and clean_file:
        noisy, _ = sf.read(noisy_file)
        clean, _ = sf.read(clean_file)

        # Truncate or pad the noisy and clean files to match the target length
        if len(noisy) > target_length:
            noisy = noisy[:target_length]
        else:
            noisy = np.pad(noisy, (0, target_length - len(noisy)), 'constant')

        if len(clean) > target_length:
            clean = clean[:target_length]
        else:
            clean = np.pad(clean, (0, target_length - len(clean)), 'constant')

        return noisy, clean

    return None, None

def create_training_data(noisy, clean, filter_order):
    """
    Create training data for the ANC model.
    
    Args:
        noisy (np.array): Array of noisy audio data.
        clean (np.array): Array of clean audio data.
        filter_order (int): The order of the adaptive filter.
        
    Returns:
        tuple: Training input and target data.
    """
    x_train = []
    y_train = []

    for i in range(filter_order, len(noisy)):
        x_train.append(noisy[i-filter_order:i])
        y_train.append(clean[i])

    return np.array(x_train), np.array(y_train)

if __name__ == "__main__":
    filter_order = 32
    learning_rate = 0.001
    epochs = 1
    batch_size = 32
    weights_path = 'anc_weights.h5'
    tflite_filename = 'anc_model.tflite'

    # Path to your root directory containing subdirectories with noisy and clean audio files
    root_directory = r'C:\Users\divya\Desktop\ANC_Project\data\artificial_data\new_training_data'

    # Find the minimum length of the audio files
    target_length = find_min_length(root_directory)

    # Initialize the ANC model
    anc_model = ANCModel(input_shape=(filter_order,), learning_rate=learning_rate)

    # Train on each subdirectory incrementally
    for subdir, _, files in os.walk(root_directory):
        if any('mixed_output.wav' in file for file in files) and any('voice_output.wav' in file for file in files):
            noisy, clean = load_and_preprocess_audio(subdir, target_length)
            if noisy is not None and clean is not None:
                x_train, y_train = create_training_data(noisy, clean, filter_order)
                anc_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)
                print(f"Trained on data from {subdir}")

    # Save the trained model weights
    anc_model.save_weights(weights_path)
    # Save the model as TFLite
    anc_model.save_tflite(tflite_filename)

    print(f"Training completed. Weights saved to {weights_path} and model saved to {tflite_filename}.")
