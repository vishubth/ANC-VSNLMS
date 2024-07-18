import logging
import os
from adaptive_filter import AdaptiveFilter
from utils import read_audio_file, write_audio_file, find_wav_files, calculate_snr, align_audio_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_new_input(filter_instance, filepath):
    """
    Test the adaptive filter on a new input file.

    Args:
        filter_instance (AdaptiveFilter): The adaptive filter instance.
        filepath (str): Path to the new input file.
    """
    try:
        noisy_signal, sr = read_audio_file(filepath)
        clean_output = filter_instance.apply_filter(noisy_signal)
        output_filepath = filepath.replace('mixed', 'filtered')
        write_audio_file(output_filepath, clean_output, sr)
        logger.info(f"Filtered output saved to {output_filepath}")
    except Exception as e:
        logger.error(f"Error in testing new input: {e}")

def process_directory(directory, adaptive_filter):
    """
    Process a directory of audio files for training the adaptive filter.

    Args:
        directory (str): Path to the root directory containing audio files.
        adaptive_filter (AdaptiveFilter): The adaptive filter instance.
    """
    try:
        for subdir, dirs, files in os.walk(directory):
            files = sorted(files)
            noisy_files = [f for f in files if 'mixed' in f]
            clean_files = [f for f in files if 'voice' in f]
            for noisy_file, clean_file in zip(noisy_files, clean_files):
                noisy_path = os.path.join(subdir, noisy_file)
                clean_path = os.path.join(subdir, clean_file)
                noisy_signal, sr = read_audio_file(noisy_path)
                clean_signal, _ = read_audio_file(clean_path)
                adaptive_filter.train(noisy_signal, clean_signal)
    except Exception as e:
        logger.error(f"Error processing directory {directory}: {e}")
        raise

def validate_filter(filter_instance, noisy_file_path, clean_file_path):
    """
    Validate the adaptive filter by comparing the SNR before and after filtering.

    Args:
        filter_instance (AdaptiveFilter): The adaptive filter instance.
        noisy_file_path (str): Path to the noisy input file.
        clean_file_path (str): Path to the clean desired file.
    """
    try:
        noisy_signal, sr = read_audio_file(noisy_file_path)
        clean_signal, _ = read_audio_file(clean_file_path)
        noisy_signal, clean_signal = align_audio_signals(noisy_signal, clean_signal)
        filtered_output = filter_instance.apply_filter(noisy_signal)
        noise = noisy_signal - clean_signal
        filtered_noise = noisy_signal - filtered_output
        original_snr = calculate_snr(clean_signal, noise)
        filtered_snr = calculate_snr(clean_signal, filtered_noise)
        snr_improvement = filtered_snr - original_snr
        logger.info(f"Original SNR: {original_snr:.2f} dB")
        logger.info(f"Filtered SNR: {filtered_snr:.2f} dB")
        logger.info(f"SNR Improvement: {snr_improvement:.2f} dB")
    except Exception as e:
        logger.error(f"Error in filter validation: {e}")

if __name__ == "__main__":
    filter_order = 32
    vsnlms_params = (0.01, 0.5, 0.0001, 10, 10, 1.1)
    weights_path = 'adaptive_filter_weights.npy'
    adaptive_filter = AdaptiveFilter(filter_order, vsnlms_params, weights_path)

    root_directory = r'C:\Users\divya\Desktop\ANC_Project\data\artificial_data\new_training_data'
    process_directory(root_directory, adaptive_filter)

    paths = find_wav_files(r"C:\Users\divya\Desktop\ANC_Project\data\artificial_data\new_test_data")
    for new_input_path in paths:
        test_new_input(adaptive_filter, new_input_path)
        validate_filter(adaptive_filter, new_input_path, new_input_path.replace('mixed_output', 'voice_output'))

    adaptive_filter.plot_performance()
