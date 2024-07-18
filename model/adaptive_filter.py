import numpy as np
import os
import random
import matplotlib.pyplot as plt
import logging
from vsnlms import VSNLMS

logger = logging.getLogger(__name__)

class AdaptiveFilter:
    """
    Adaptive filter using the VSNLMS algorithm.

    Attributes:
        filter_order (int): The order of the filter.
        vsnlms (VSNLMS): The VSNLMS filter instance.
        total_error (float): Sum of squared errors for training.
        num_samples (int): Number of samples processed.
        rmse_history (list): History of RMSE values.
        weights_path (str): Path to save/load filter weights.
    """

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
        """
        Train the adaptive filter using noisy and clean signals.

        Args:
            noisy_signal (np.array): The noisy input signal.
            clean_signal (np.array): The clean desired signal.
        """
        try:
            N = min(len(noisy_signal), len(clean_signal))
            batch_id = random.randint(1000, 99999)
            self.total_error = 0
            self.num_samples = 0
            for n in range(self.filter_order, N):
                x = noisy_signal[n-self.filter_order:n]
                if len(x.shape) > 1:
                    x = x.ravel()
                d = clean_signal[n]
                error, weights = self.vsnlms.update_weights(x, d)
                self.total_error += error ** 2
                self.num_samples += 1
            if self.weights_path:
                self.save_weights(self.weights_path)

            average_error = (self.total_error / self.num_samples) ** 0.5 if self.num_samples > 0 else float('inf')
            self.rmse_history.append(average_error)
            logger.info('*'*50)
            logger.info(f"Batch ID: {batch_id} - Training completed with {self.num_samples} iterations.")
            logger.info(f"Root Mean Square Error (RMSE): {average_error:.4f}")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def apply_filter(self, input_signal):
        """
        Apply the adaptive filter to an input signal.

        Args:
            input_signal (np.array): The input signal to be filtered.

        Returns:
            np.array: The filtered output signal.
        """
        try:
            if self.weights_path and os.path.exists(self.weights_path):
                self.load_weights(self.weights_path)
            N = len(input_signal)
            output = np.zeros(N)
            for n in range(self.filter_order, N):
                x = input_signal[n-self.filter_order:n][::-1]
                output[n] = np.dot(self.vsnlms.weights, x)
            return output
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            raise

    def save_weights(self, filename):
        """
        Save the current filter weights to a file.

        Args:
            filename (str): The path to the file where weights will be saved.
        """
        try:
            np.save(filename, self.vsnlms.weights)
        except Exception as e:
            logger.error(f"Error saving weights to {filename}: {e}")
            raise

    def load_weights(self, filename):
        """
        Load filter weights from a file.

        Args:
            filename (str): The path to the file from which weights will be loaded.
        """
        try:
            self.vsnlms.weights = np.load(filename)
        except Exception as e:
            logger.error(f"Error loading weights from {filename}: {e}")
            raise

    def plot_performance(self):
        """
        Plot the RMSE performance over training batches.
        """
        try:
            if self.rmse_history:
                plt.figure(figsize=(10, 5))
                plt.plot(self.rmse_history, label='RMSE over Batches')
                plt.title('RMSE Convergence Over Training Batches')
                plt.xlabel('Batch Number')
                plt.ylabel('Root Mean Square Error')
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                logger.warning("No RMSE data to plot.")
        except Exception as e:
            logger.error(f"Error plotting performance: {e}")
            raise
