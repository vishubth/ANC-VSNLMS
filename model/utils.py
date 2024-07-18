import numpy as np
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

def read_audio_file(filename):
    """
    Read an audio file and return the normalized data and sample rate.

    Args:
        filename (str): Path to the audio file.

    Returns:
        tuple: Normalized audio data and sample rate.
    """
    try:
        data, samplerate = sf.read(filename)
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        max_val = np.max(np.abs(data))
        if max_val == 0:
            logger.warning(f"Audio data in file {filename} is silent or zeroed out.")
            normalized_data = data
        else:
            normalized_data = data / max_val
        return normalized_data, samplerate
    except Exception as e:
        logger.error(f"Failed to read audio file {filename}: {e}")
        raise

def write_audio_file(filename, data, samplerate):
    """
    Write audio data to a file.

    Args:
        filename (str): Path to the output audio file.
        data (np.array): Audio data to be written.
        samplerate (int): Sample rate of the audio data.
    """
    try:
        sf.write(filename, data, samplerate)
    except Exception as e:
        logger.error(f"Failed to write audio file {filename}: {e}")
        raise

def calculate_snr(signal, noise):
    """
    Calculate the Signal to Noise Ratio (SNR).

    Args:
        signal (np.array): Clean signal.
        noise (np.array): Noise signal.

    Returns:
        float: SNR in decibels.
    """
    try:
        signal_power = np.mean(np.square(signal))
        noise_power = np.mean(np.square(noise))
        if noise_power == 0:
            return float('inf')
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    except Exception as e:
        logger.error(f"Error calculating SNR: {e}")
        raise

def align_audio_signals(signal_a, signal_b):
    """
    Align two audio signals to the same length by trimming.

    Args:
        signal_a (np.array): First audio signal.
        signal_b (np.array): Second audio signal.

    Returns:
        tuple: Aligned audio signals.
    """
    try:
        min_length = min(len(signal_a), len(signal_b))
        signal_a_aligned = signal_a[:min_length]
        signal_b_aligned = signal_b[:min_length]
        return signal_a_aligned, signal_b_aligned
    except Exception as e:
        logger.error(f"Error aligning audio signals: {e}")
        raise

def find_wav_files(directory, filename='mixed_output.wav'):
    """
    Find all .wav files in a directory and its subdirectories.

    Args:
        directory (str): The root directory to search.
        filename (str): The specific filename to look for.

    Returns:
        list: List of paths to the found .wav files.
    """
    try:
        wav_files = []
        for root, dirs, files in os.walk(directory):
            if filename in files:
                wav_files.append(os.path.join(root, filename))
        return wav_files
    except Exception as e:
        logger.error(f"Error finding .wav files in directory {directory}: {e}")
        raise
