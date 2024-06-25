# ANC-VSNLMS
Real Time Active Noise Cancellation with VNSLMS Algorithm

# AdaptiveFilter-based ANC System

This repository contains the implementation of an AdaptiveFilter-based Active Noise Cancellation (ANC) system using the Variable Step-size Normalized Least Mean Squares (VSNLMS) algorithm. The system is designed to effectively reduce ambient noise in audio signals, making it suitable for various applications such as audio processing and communication enhancement.

## Features

- **Adaptive Noise Cancellation**: Utilizes dynamic filter coefficients to minimize ambient noise in real-time.
- **Variable Step-size Algorithm**: Implements the VSNLMS algorithm, which adjusts step size based on environmental noise changes.
- **Persistent State Management**: Supports saving and loading of the adaptive filter coefficients, enabling effective state management and ease of deployment.

## Installation

### Prerequisites

- Conda (Anaconda or Miniconda)

### Environment Setup

To ensure consistent setup across any platform, we use a Conda environment. Follow these steps:

1. Clone the repository:
   git clone https://github.com/your-username/adaptivefilter-anc.git
   cd adaptivefilter-anc

2. Create the Conda environment from the environment.yml file:
    conda env create -f environment.yml

3. Activate the newly created environment:
    conda activate anc


### Usage
Ensure you're in the Conda environment (adaptivefilter-anc) before running these commands.

1. Place your noisy and reference clean audio files in an appropriate directory (e.g., ./audio).

2. To run the ANC system and process audio files:

    python adaptive_filter.py --noisy path/to/noisy_file.wav --clean path/to/clean_file.wav --output path/to/output_file.wav


3. To save the current filter weights after training:
    python adaptive_filter.py --save-weights path/to/weights_file.npy


4. To load pre-existing weights for immediate use without re-training:

    python adaptive_filter.py --load-weights path/to/weights_file.npy


### License
This project is licensed under the Apache License - see the LICENSE file for details.


### Notes:
- Ensure that the actual paths and command-line options match those implemented in your Python scripts.
- If your project does not yet include a `LICENSE` file and you wish to use the MIT License, be sure to add one.
- This README assumes the use of a command-line interface for simplicity and broad applicability. Adjust the usage instructions if your application has a graphical interface or other specific usage scenarios.

