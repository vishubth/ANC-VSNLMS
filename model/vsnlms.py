import numpy as np

class VSNLMS:
    """
    Variable Step-size Normalized Least Mean Squares (VSNLMS) adaptive filter.

    Attributes:
        mu (float): Initial step size.
        mu_max (float): Maximum step size.
        mu_min (float): Minimum step size.
        m0, m1 (int): Parameters for step size adjustment.
        alpha (float): Step size adjustment factor.
        delta (float): Error threshold for adjusting step size.
        weights (np.array): Filter weights.
    """
    
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
        """
        Adjusts the step size based on the error magnitude.

        Args:
            error (float): The current error signal.
        """
        if abs(error) > self.delta:
            self.mu = min(self.mu * self.alpha, self.mu_max)
        else:
            self.mu = max(self.mu / self.alpha, self.mu_min)

    def update_weights(self, x, d):
        """
        Updates the filter weights based on the input and desired output.

        Args:
            x (np.array): Input signal segment.
            d (float): Desired output.

        Returns:
            tuple: Current error and updated weights.
        """
        y = np.dot(x, self.weights)
        error = d - y
        self.adjust_step_size(error)

        norm_factor = np.dot(x, x) + 0.01
        self.weights += self.mu * error * x / norm_factor

        weight_norm = np.linalg.norm(self.weights)
        if weight_norm > 10:
            self.weights /= weight_norm

        return error, self.weights.copy()
