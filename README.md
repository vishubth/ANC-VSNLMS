# ANC-VSNLMS

Real-time Active Noise Cancellation (ANC) system implementing a Variable Step-size Normalized Least Mean Squares (VSNLMS) adaptive filtering pipeline for dynamic acoustic environments.

---

# Overview

This project implements an adaptive signal processing system designed to suppress environmental noise from audio streams using a dynamically updating adaptive filter.

Unlike static filtering approaches, the system continuously adjusts filter coefficients in response to changing acoustic conditions. This makes it suitable for non-stationary noise environments where traditional fixed filters perform poorly.

The implementation was developed as part of real client-oriented engineering work involving:

- adaptive audio processing
- low-latency filtering pipelines
- real-time coefficient adaptation
- noise minimization under changing signal conditions
- experimentation with convergence stability and dynamic learning rates

---

# Core Objective

The goal of the system is to minimize the residual error signal between:

- the noisy primary input signal
- and the estimated environmental noise signal

by dynamically updating filter coefficients using the VSNLMS adaptive optimization algorithm.

---

# Why VSNLMS?

Traditional LMS filters often suffer from:

- slow convergence
- instability under changing signal amplitudes
- sensitivity to input scaling
- poor adaptation in non-stationary environments

The Variable Step-size Normalized Least Mean Squares (VSNLMS) approach was selected because it:

- normalizes updates against input power
- improves convergence stability
- dynamically adjusts adaptation rate
- balances convergence speed and steady-state error
- performs better under changing environmental noise profiles

This makes VSNLMS significantly more robust for real-time ANC applications.

---

# System Architecture

```text
                +-------------------+
                |  Reference Noise  |
                +---------+---------+
                          |
                          v
                  +---------------+
                  | VSNLMS Filter |
                  +-------+-------+
                          |
                          v
+-------------+    +-------------+    +----------------+
| Primary Mic | -> | Error Signal| -> | Noise Reduced  |
|  (Noise +   |    | Minimization|    | Output Audio   |
|   Desired)  |    +-------------+    +----------------+
+-------------+
```

---

# Signal Processing Pipeline

1. Capture noisy input signal
2. Capture reference environmental noise signal
3. Pass reference signal through adaptive filter
4. Estimate correlated noise component
5. Subtract estimated noise from primary signal
6. Compute residual error signal
7. Update adaptive filter coefficients using VSNLMS
8. Produce noise-suppressed output

---

# Technical Highlights

## Adaptive Filtering

Implements continuously updating filter coefficients to adapt to changing acoustic environments in real time.

## Dynamic Step-size Optimization

The learning rate adapts based on signal characteristics to improve both convergence speed and long-term stability.

## Persistent State Management

Supports serialization and restoration of learned filter weights for deployment continuity and experimentation.

## Audio Processing Tooling

Includes support libraries for:

- waveform analysis
- preprocessing
- experimentation
- optimization workflows
- signal visualization

---

# Engineering Challenges Addressed

This project involved solving several practical DSP and systems-engineering challenges:

## Convergence Stability

Balancing aggressive adaptation against oscillation and instability during rapid environmental changes.

## Non-Stationary Noise

Handling continuously changing ambient noise patterns rather than static synthetic datasets.

## Real-Time Constraints

Maintaining low-latency processing while continuously updating adaptive coefficients.

## Signal Power Normalization

Preventing divergence caused by amplitude variation in incoming audio streams.

## Filter Generalization

Designing a system capable of adapting across different acoustic environments rather than overfitting to a single noise profile.

---

# Technology Stack

## Core Libraries

- Python
- NumPy
- SciPy
- Librosa
- SoundDevice
- SoundFile
- Matplotlib

## Experimentation / Optimization

- TensorFlow
- Keras
- Optuna
- TensorBoard

## Audio & DSP Tooling

- PyDub
- noisereduce
- librosa

---

# Installation

## Prerequisites

- Conda (Anaconda or Miniconda)
- Python 3.8+

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/vishubth/ANC-VSNLMS.git
cd ANC-VSNLMS
```

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate anc
```

---

# Usage

## Run ANC Processing

```bash
python adaptive_filter.py \
    --noisy path/to/noisy.wav \
    --clean path/to/reference.wav \
    --output path/to/output.wav
```

## Save Learned Filter Weights

```bash
python adaptive_filter.py \
    --save-weights path/to/weights.npy
```

## Load Existing Filter Weights

```bash
python adaptive_filter.py \
    --load-weights path/to/weights.npy
```

---

# Recommended Future Improvements

## Benchmarking

Potential evaluation metrics:

- Signal-to-Noise Ratio (SNR)
- Mean Squared Error (MSE)
- Convergence Time
- Latency Measurements
- Frequency-domain Analysis

## Visualization Enhancements

Recommended additions:

- waveform comparison plots
- spectrogram analysis
- convergence graphs
- filter coefficient evolution plots

## Real-Time Deployment

Potential deployment targets:

- embedded audio hardware
- edge AI systems
- communication devices
- live audio processing pipelines

---



This repository demonstrates experience with:

- adaptive signal processing
- numerical optimization
- DSP algorithm implementation
- real-time systems engineering
- audio processing pipelines
- low-level mathematical programming
- experimentation workflows
- client-oriented engineering delivery

---

# License

This project is licensed under the Apache License.
