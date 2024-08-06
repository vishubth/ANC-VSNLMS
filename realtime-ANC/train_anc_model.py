# --------------------------------------------------------------------------------------------------------------
# 
# Author - Vishal Shrivastava
# 
# ---------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import soundfile as sf
from anc_model import ANCModel
from preprocessing import preprocess_audio_data, create_training_data

def pad_or_trim(audio, target_length):
    """
    Pad or trim audio data to a target length.
    
    Args:
        audio (np.array): Input audio data.
        target_length (int): Target length for the audio data.
    
    Returns:
        np.array: Audio data with the specified target length.
    """
    if len(audio) > target_length:
        return audio[:target_length]
    else:
        return np.pad(audio, (0, target_length - len(audio)), 'constant')

def load_audio_files(noisy_files, clean_files, target_length=None):
    """
    Load audio files for training and ensure consistent length.
    
    Args:
        noisy_files (list): List of paths to noisy audio files.
        clean_files (list): List of paths to clean audio files.
        target_length (int): Target length for padding/truncating audio files.
    
    Returns:
        tuple: Arrays of noisy and clean audio data with consistent length.
    """
    noisy_data = []
    clean_data = []

    for noisy_file, clean_file in zip(noisy_files, clean_files):
        noisy, _ = sf.read(noisy_file)
        clean, _ = sf.read(clean_file)

        if target_length:
            # Ensure the audio files have consistent length
            noisy = pad_or_trim(noisy, target_length)
            clean = pad_or_trim(clean, target_length)

        noisy_data.append(noisy)
        clean_data.append(clean)

    return np.array(noisy_data), np.array(clean_data)

def main():
    filter_order = 32
    learning_rate = 0.001
    epochs = 10
    batch_size = 32
    weights_path = 'anc_weights.h5'
    tflite_filename = 'anc_model.tflite'
    target_length = 16000  # Set to desired target length

    # Paths to your noisy and clean audio files
    root_directory = r'C:\Users\divya\Desktop\ANC_Project\data\artificial_data\new_training_data'
    noisy_files = []
    clean_files = []

    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if 'mixed_output' in file:
                noisy_files.append(os.path.join(subdir, file))
            elif 'voice_output' in file:
                clean_files.append(os.path.join(subdir, file))

    # Ensure noisy_files and clean_files are sorted to match pairs
    noisy_files = sorted(noisy_files)
    clean_files = sorted(clean_files)

    # Load audio files
    noisy_data, clean_data = load_audio_files(noisy_files, clean_files, target_length)

    # Preprocess audio data
    preprocessed_noisy_data, preprocessed_clean_data = preprocess_audio_data(noisy_data, clean_data, target_length)

    # Create training data
    x_train, y_train = create_training_data(preprocessed_noisy_data, preprocessed_clean_data, filter_order)

    # Initialize and train the ANC model
    anc_model = ANCModel(input_shape=(filter_order, 1), learning_rate=learning_rate)
    anc_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the trained model weights
    anc_model.save_weights(weights_path)

    # Save the model as TFLite
    anc_model.save_tflite(tflite_filename)

    print(f"Training completed. Weights saved to {weights_path} and model saved to {tflite_filename}.")

if __name__ == "__main__":
    main()
