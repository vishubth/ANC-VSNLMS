import numpy as np

def normalize_audio(audio_data):
    """
    Normalize audio data to have zero mean and unit variance.
    
    Args:
        audio_data (np.array): Array of audio data.
    
    Returns:
        np.array: Normalized audio data.
    """
    mean = np.mean(audio_data)
    std = np.std(audio_data)
    if std == 0:
        return audio_data
    return (audio_data - mean) / std

def preprocess_audio_data(noisy_data, clean_data, target_length=None):
    """
    Preprocess noisy and clean audio data by normalizing and adjusting length.
    
    Args:
        noisy_data (np.array): Array of noisy audio data.
        clean_data (np.array): Array of clean audio data.
        target_length (int): Target length for padding/truncating audio files.
    
    Returns:
        tuple: Preprocessed noisy and clean audio data.
    """
    preprocessed_noisy_data = []
    preprocessed_clean_data = []

    for noisy, clean in zip(noisy_data, clean_data):
        if target_length is None:
            target_length = min(len(noisy), len(clean))

        # Trim or pad to target length
        if len(noisy) > target_length:
            noisy = noisy[:target_length]
        else:
            noisy = np.pad(noisy, (0, target_length - len(noisy)), 'constant')

        if len(clean) > target_length:
            clean = clean[:target_length]
        else:
            clean = np.pad(clean, (0, target_length - len(clean)), 'constant')

        # Normalize the audio data
        noisy = normalize_audio(noisy)
        clean = normalize_audio(clean)

        preprocessed_noisy_data.append(noisy)
        preprocessed_clean_data.append(clean)

    return np.array(preprocessed_noisy_data), np.array(preprocessed_clean_data)

def create_training_data(noisy_data, clean_data, filter_order):
    """
    Create training data for the ANC model by framing the audio data into overlapping windows.
    
    Args:
        noisy_data (np.array): Array of preprocessed noisy audio data.
        clean_data (np.array): Array of preprocessed clean audio data.
        filter_order (int): The number of past samples to consider (window size).
    
    Returns:
        tuple: Training input data (x_train) and training target data (y_train).
    """
    x_train = []
    y_train = []

    for noisy, clean in zip(noisy_data, clean_data):
        for i in range(filter_order, len(noisy)):
            x_train.append(noisy[i-filter_order:i])
            y_train.append(clean[i])

    return np.array(x_train), np.array(y_train)
