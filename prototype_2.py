import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
import random
import optuna

# VSNLMS and AdaptiveFilter classes
# Include these classes from your previous script

class VSNLMS:
    def __init__(self, mu, mu_max, mu_min, m0, m1, alpha, delta=0.0):
        self.mu = mu
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.m0 = m0
        self.m1 = m1
        self.alpha = alpha
        self.delta = delta
        self.weights = None

    def adjust_step_size(self, error):
        if abs(error) > self.delta:
            self.mu = min(self.mu * self.alpha, self.mu_max)
        else:
            self.mu = max(self.mu / self.alpha, self.mu_min)

    def update_weights(self, x, d):
        y = np.dot(x, self.weights)
        error = d - y
        self.adjust_step_size(error)
        
        # Calculate the normalization factor, ensuring no division by zero
        norm_factor = np.dot(x, x) + 0.01
        
        # Update weights based on the derived error and learning rate
        self.weights += self.mu * error * x / norm_factor
        
        # Optionally normalize weights if they grow too large
        weight_norm = np.linalg.norm(self.weights)
        if weight_norm > 10:  # Adjust this threshold based on empirical observations
            self.weights /= weight_norm  # Normalize weights to keep them bounded

        return error, self.weights.copy()

class AdaptiveFilter:
    def __init__(self, filter_order, vsnlms_params, weights_path=None):
        self.filter_order = filter_order
        self.vsnlms = VSNLMS(*vsnlms_params)
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            self.vsnlms.weights = np.zeros(filter_order)
        self.total_error = 0
        self.num_samples = 0
        self.rmse_history = []
        self.weights_path = weights_path

    def train(self, noisy_signal, clean_signal):
        N = min(len(noisy_signal), len(clean_signal))
        batch_id = random.randint(1000, 99999)  # Generate a random batch ID between 1000 and 99999
        self.total_error = 0
        self.num_samples = 0
        for n in range(self.filter_order, N):
            x = noisy_signal[n-self.filter_order:n]  # Ensure x is correctly sliced
            if len(x.shape) > 1:
                x = x.ravel()  # Flatten the array if it's not already one-dimensional
            d = clean_signal[n]
            error, weights = self.vsnlms.update_weights(x, d)
            # Update running statistics
            self.total_error += error ** 2
            self.num_samples += 1
        if self.weights_path:
            self.save_weights(self.weights_path)
        
        # Calculate average error
        average_error = (self.total_error / self.num_samples) ** 0.5 if self.num_samples > 0 else float('inf')
        self.rmse_history.append(average_error)  # Append RMSE of this batch
        # Print training completion message with details
        print('*'*50)
        print(f"Batch ID: {batch_id} - Training completed with {self.num_samples} iterations.")
        print(f"Root Mean Square Error (RMSE): {average_error:.4f}")

    def apply_filter(self, input_signal):
        if self.weights_path and os.path.exists(self.weights_path):
            self.load_weights(self.weights_path)
        N = len(input_signal)
        output = np.zeros(N)
        for n in range(self.filter_order, N):
            x = input_signal[n-self.filter_order:n][::-1]
            output[n] = np.dot(self.vsnlms.weights, x)
        return output

    def save_weights(self, filename):
        np.save(filename, self.vsnlms.weights)

    def load_weights(self, filename):
        self.vsnlms.weights = np.load(filename)

    def plot_performance(self):
        if self.rmse_history:  # Check if there is RMSE data to plot
            plt.figure(figsize=(10, 5))
            plt.plot(self.rmse_history, label='RMSE over Batches')
            plt.title('RMSE Convergence Over Training Batches')
            plt.xlabel('Batch Number')
            plt.ylabel('Root Mean Square Error')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No RMSE data to plot.")

def read_audio_file(filename):
    data, samplerate = sf.read(filename)
    if len(data.shape) == 2:  # Check if the audio is stereo
        data = np.mean(data, axis=1)  # Convert stereo to mono by averaging both channels
    # Normalize the input
    max_val = np.max(np.abs(data))
    if max_val == 0:  # Check if the maximum value is zero to avoid division by zero
        print(f"Warning: Audio data in file {filename} is silent or zeroed out.")
        normalized_data = data  # Return the data as-is or handle according to your needs
    else:
        normalized_data = data / max_val  # Normalize the data only if max_val is not zero
    return normalized_data, samplerate

def write_audio_file(filename, data, samplerate):
    sf.write(filename, data, samplerate)

def test_new_input(filter_instance, filepath):
    noisy_signal, sr = read_audio_file(filepath)
    clean_output = filter_instance.apply_filter(noisy_signal)
    output_filepath = filepath.replace('mixed', 'filtered')
    write_audio_file(output_filepath, clean_output, sr)
    print(f"Filtered output saved to {output_filepath}")

def process_directory(directory, adaptive_filter):
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

def calculate_snr(signal, noise):
    """
    Calculate the Signal to Noise Ratio (SNR) given a signal and its noise.
    Both signal and noise should be numpy arrays of the same length.
    
    Args:
    signal (np.array): The clean, desired signal.
    noise (np.array): The noise in the signal, calculated as the difference between the noisy and clean signals.

    Returns:
    float: SNR in decibels.
    """
    # Calculate the power of the signal and the noise
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    if noise_power == 0:
        return float('inf')  # Avoid division by zero
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def align_audio_signals(signal_a, signal_b):
    """
    Align two audio signals in length by trimming the longer or padding the shorter with zeros.
    
    Args:
    signal_a (np.array): First audio signal.
    signal_b (np.array): Second audio signal.

    Returns:
    tuple: Tuple of numpy arrays (signal_a_aligned, signal_b_aligned)
    """
    min_length = min(len(signal_a), len(signal_b))
    # Trim both signals to the minimum length of either signal
    signal_a_aligned = signal_a[:min_length]
    signal_b_aligned = signal_b[:min_length]
    return signal_a_aligned, signal_b_aligned

def validate_filter(filter_instance, noisy_file_path, clean_file_path):
    noisy_signal, sr = read_audio_file(noisy_file_path)
    clean_signal, _ = read_audio_file(clean_file_path)

    # Align the lengths of both signals
    noisy_signal, clean_signal = align_audio_signals(noisy_signal, clean_signal)
    filtered_output = filter_instance.apply_filter(noisy_signal)
    
    # Calculate the noise as the difference between noisy input and clean signal
    noise = noisy_signal - clean_signal
    filtered_noise = noisy_signal - filtered_output

    original_snr = calculate_snr(clean_signal, noise)
    filtered_snr = calculate_snr(clean_signal, filtered_noise)
    
    snr_improvement = filtered_snr - original_snr
    print(f"Original SNR: {original_snr:.2f} dB")
    print(f"Filtered SNR: {filtered_snr:.2f} dB")
    print(f"SNR Improvement: {snr_improvement:.2f} dB")

def objective(trial):
    # Define the search space for each parameter
    mu = trial.suggest_loguniform('mu', 1e-4, 1e-2)
    mu_max = trial.suggest_loguniform('mu_max', 1e-2, 1.0)
    mu_min = trial.suggest_loguniform('mu_min', 1e-5, 1e-2)
    m0 = trial.suggest_int('m0', 10, 30)
    m1 = trial.suggest_int('m1', 10, 30)
    alpha = trial.suggest_uniform('alpha', 1.01, 1.2)

    vsnlms_params = (mu, mu_max, mu_min, m0, m1, alpha)
    adaptive_filter = AdaptiveFilter(filter_order, vsnlms_params, weights_path)

    # Training phase
    process_directory(root_directory, adaptive_filter)

    # Validation phase on a new input
    new_input_path = r"C:\Users\divya\Desktop\ANC_Project\data\artificial_data\Final2\output_sample_44\mixed_output.wav"
    validate_filter(adaptive_filter, new_input_path, new_input_path.replace('mixed_output', 'voice_output'))
    
    # Return the validation RMSE as the objective value
    return adaptive_filter.rmse_history[-1]

if __name__ == "__main__":
    filter_order = 32  # 64 og
    weights_path = 'adaptive_filter_weights.npy'
    root_directory = r'C:\Users\divya\Desktop\ANC_Project\data\artificial_data\Final2'

    # Optimize the parameters using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed

    print(f"Best parameters: {study.best_params}")
    print(f"Best RMSE: {study.best_value}")

