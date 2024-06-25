import numpy as np
import soundfile as sf

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
        """ Adjusts the step size dynamically based on error conditions. """
        if abs(error) > self.delta:
            self.mu = min(self.mu * self.alpha, self.mu_max)
        else:
            self.mu = max(self.mu / self.alpha, self.mu_min)

    def update_weights(self, x, d):
        """ Update filter weights based on input vector x and desired output d. """
        y = np.dot(x, self.weights)
        error = d - y
        self.adjust_step_size(error)
        norm_factor = np.dot(x, x) + 0.01  # Avoid division by zero
        self.weights += self.mu * error * x / norm_factor
        return error

class AdaptiveFilter:
    def __init__(self, filter_order, vsnlms_params):
        self.filter_order = filter_order
        self.vsnlms = VSNLMS(*vsnlms_params)
        self.vsnlms.weights = np.zeros(filter_order)

    def train(self, noisy_signal, clean_signal):
        N = len(noisy_signal)
        for n in range(self.filter_order, N):
            x = noisy_signal[n-self.filter_order:n][::-1]
            d = clean_signal[n]
            self.vsnlms.update_weights(x, d)

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

def read_audio_file(filename):
    data, samplerate = sf.read(filename)
    return data, samplerate

def write_audio_file(filename, data, samplerate):
    sf.write(filename, data, samplerate)

if __name__ == "__main__":
    filter_order = 64
    vsnlms_params = (0.01, 0.5, 0.0001, 10, 10, 1.1)  # mu, mu_max, mu_min, m0, m1, alpha

    adaptive_filter = AdaptiveFilter(filter_order, vsnlms_params)
    noisy_signal, sr = read_audio_file('path/to/noisy_voice.wav')
    clean_signal, _ = read_audio_file('path/to/clean_voice.wav')

    adaptive_filter.train(noisy_signal, clean_signal)

    filtered_output = adaptive_filter.apply_filter(noisy_signal)
    write_audio_file('path/to/filtered_voice.wav', filtered_output, sr)

    # Save and load weight functionality
    adaptive_filter.save_weights('adaptive_filter_weights.npy')
    adaptive_filter.load_weights('adaptive_filter_weights.npy')
