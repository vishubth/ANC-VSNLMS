# --------------------------------------------------------------------------------------------------------------
# 
# Author - Vishal Shrivastava
# 
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.signal import butter, lfilter

def normalize_output(output_data, max_value=1.0):
    """
    Normalize the output audio data to be within the range [-max_value, max_value].
    
    Args:
        output_data (np.array): Array of output audio data from the model.
        max_value (float): Maximum absolute value for normalization.
    
    Returns:
        np.array: Normalized output audio data.
    """
    max_amplitude = np.max(np.abs(output_data))
    if max_amplitude > 0:
        output_data = (output_data / max_amplitude) * max_value
    return output_data

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut (float): Low frequency cut-off.
        highcut (float): High frequency cut-off.
        fs (float): Sampling frequency.
        order (int): Order of the filter.
    
    Returns:
        tuple: Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    print(f"Nyquist Frequency: {nyquist}")
    print(f"Normalized Lowcut: {low}")
    print(f"Normalized Highcut: {high}")

    if not 0 < low < 1 or not 0 < high < 1:
        raise ValueError(f"Digital filter critical frequencies must be 0 < Wn < 1. Received low: {low}, high: {high}")

    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the data.
    
    Args:
        data (np.array): Input audio data.
        lowcut (float): Low frequency cut-off.
        highcut (float): High frequency cut-off.
        fs (float): Sampling frequency.
        order (int): Order of the filter.
    
    Returns:
        np.array: Bandpass filtered audio data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def reduce_noise(output_data, noise_level=0.01):
    """
    Apply simple noise reduction by thresholding.
    
    Args:
        output_data (np.array): Array of output audio data from the model.
        noise_level (float): Threshold below which values will be set to zero.
    
    Returns:
        np.array: Noise-reduced output audio data.
    """
    return np.where(np.abs(output_data) < noise_level, 0, output_data)
