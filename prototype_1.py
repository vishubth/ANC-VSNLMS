import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
import random

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
        norm_factor = np.dot(x, x) + 0.01  # Avoid division by zero
        self.weights += self.mu * error * x / norm_factor
        return error, self.weights.copy()

class AdaptiveFilter:
    def __init__(self, filter_order, vsnlms_params, weights_path=None):
        self.filter_order = filter_order
        self.vsnlms = VSNLMS(*vsnlms_params)
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            self.vsnlms.weights = np.zeros(filter_order)
        self.errors = []
        self.weight_history = []
        self.weights_path = weights_path

    def train(self, noisy_signal, clean_signal):
        N = min(len(noisy_signal), len(clean_signal))
        batch_id = random.randint(1000, 99999)  # Generate a random batch ID between 1000 and 99999
        for n in range(self.filter_order, N):
            x = noisy_signal[n-self.filter_order:n]  # Ensure x is correctly sliced
            if len(x.shape) > 1:
                x = x.ravel()  # Flatten the array if it's not already one-dimensional
            d = clean_signal[n]
            error, weights = self.vsnlms.update_weights(x, d)
            self.errors.append(error)
            self.weight_history.append(weights)
        if self.weights_path:
            self.save_weights(self.weights_path)
        # Print training completion message with details
        print('*'*50)
        print(f"Batch ID: {batch_id} - Training completed with {N-self.filter_order} iterations.")
        print(f"Last error: {self.errors[-1]:.4f}")
        if self.errors:
            print(f"Average error: {np.mean(self.errors):.4f}")
        if self.weight_history:
            print(f"Final weight snapshot: {self.weight_history[-1]}")

    def apply_filter(self, input_signal):
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
        plt.figure(figsize=(12, 6))
        
        # Error convergence plot
        plt.subplot(1, 2, 1)
        # Subsample the errors if too many points, plot every 10th point if too many
        step = len(self.errors) // 5000 + 1  # Adjust the step to ensure we plot at most 5000 points
        plt.plot(self.errors[::step], label='Error over Iterations')
        plt.title('Error Convergence')
        plt.xlabel(f'Iterations (every {step}th)')
        plt.ylabel('Error')
        plt.grid(True)

        # Weights convergence plot
        plt.subplot(1, 2, 2)
        weights_array = np.array(self.weight_history[::step])  # Subsample for plotting
        for i in range(weights_array.shape[1]):
            plt.plot(weights_array[:, i], label=f'Weight {i+1}')
        plt.title('Filter Weights Convergence')
        plt.xlabel(f'Iterations (every {step}th)')
        plt.ylabel('Weight Value')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


def read_audio_file(filename):
    data, samplerate = sf.read(filename)
    if len(data.shape) == 2:  # Check if the audio is stereo
        data = np.mean(data, axis=1)  # Convert stereo to mono by averaging both channels
    return data, samplerate


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

if __name__ == "__main__":
    filter_order = 64
    vsnlms_params = (0.01, 0.5, 0.0001, 10, 10, 1.1)  # mu, mu_max, mu_min, m0, m1, alpha
    weights_path = 'adaptive_filter_weights.npy'
    adaptive_filter = AdaptiveFilter(filter_order, vsnlms_params, weights_path)

    # Training phase
    root_directory = r'C:\Users\divya\Desktop\ANC_Project\data\artificial_data\Final'
    process_directory(root_directory, adaptive_filter)

    # Testing phase on new input
    new_input_path = r"C:\Users\divya\Desktop\ANC_Project\data\artificial_data\Mixed\output_Speaker_0019_00014_157207-6-9-0\mixed_output.wav"
    test_new_input(adaptive_filter, new_input_path)

    # Optionally, plot the performance after all trainings
    adaptive_filter.plot_performance()
