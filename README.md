# ANC-VSNLMS

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![DSP](https://img.shields.io/badge/domain-adaptive%20DSP-green)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)
![License](https://img.shields.io/badge/license-MIT-yellow)

Modular adaptive signal-processing platform implementing Variable Step-size Normalized Least Mean Squares (VSNLMS) adaptive filtering for Active Noise Cancellation (ANC) in dynamic acoustic environments.

---

# Overview

ANC-VSNLMS is a production-oriented adaptive DSP experimentation platform focused on:

- real-time Active Noise Cancellation
- adaptive filtering
- convergence optimization
- numerical stability
- DSP benchmarking
- runtime audio processing
- evaluation and visualization pipelines

The repository evolved from real client-oriented DSP experimentation into a modular engineering-focused signal-processing system.

---

# System Architecture

```text
Audio Input
    ↓
Realtime Runtime Pipeline
    ↓
VSNLMS Adaptive Filter
    ↓
Error Minimization
    ↓
Benchmark Evaluation
    ↓
Visualization + Analysis
```

---

# Repository Structure

```text
src/
├── filters/
│   ├── vsnlms.py
│   └── adaptive_filter.py
├── realtime/
│   └── live_audio_filter.py
├── evaluation/
│   ├── metrics.py
│   └── benchmark_runner.py
├── visualization/
│   ├── waveform_visualizer.py
│   ├── spectrogram_visualizer.py
│   └── convergence_plots.py
└── __init__.py
```

---

# Technical Highlights

## Adaptive Filtering

Implements dynamic coefficient updates for changing acoustic environments.

## VSNLMS Optimization

Uses normalized adaptive updates with variable step-size control to improve convergence stability and runtime robustness.

## DSP Benchmarking

Includes evaluation tooling for:

- SNR analysis
- RMSE analysis
- convergence tracking
- benchmark automation

## Visualization Infrastructure

Supports:

- waveform visualization
- spectrogram analysis
- convergence plotting
- DSP result inspection

## Runtime Processing

Includes realtime-oriented ANC runtime orchestration for live audio experimentation.

---

# Engineering Challenges Addressed

- adaptive convergence stability
- non-stationary noise handling
- runtime DSP orchestration
- coefficient divergence prevention
- signal normalization
- realtime filtering constraints
- low-latency processing

---

# Installation

## Clone Repository

```bash
git clone https://github.com/vishubth/ANC-VSNLMS.git
cd ANC-VSNLMS
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Run Realtime ANC Pipeline

```bash
python examples/run_realtime_anc.py
```

## Run Benchmark Evaluation

```bash
python examples/run_benchmark.py
```

---

# Benchmarking Pipeline

The repository includes benchmarking support for:

- Signal-to-Noise Ratio (SNR)
- Root Mean Square Error (RMSE)
- convergence evaluation
- DSP runtime analysis

Benchmark execution:

```bash
python examples/run_benchmark.py
```

---

# Visualization Pipeline

Visualization modules can generate:

- waveform comparisons
- spectrogram analysis
- convergence graphs

Generated artifacts should be stored under:

```text
assets/
├── waveforms/
├── spectrograms/
├── benchmarks/
└── samples/
```

---

# CI/CD

GitHub Actions workflow includes:

- automated testing
- lint validation
- package validation
- multi-version Python execution

---

# Interview-Relevant Engineering Areas

This repository demonstrates practical engineering work involving:

- adaptive signal processing
- numerical optimization
- realtime DSP systems
- runtime audio pipelines
- benchmarking infrastructure
- modular software architecture
- experimentation workflows
- convergence stabilization
- production-oriented Python engineering

---

# License

This project is licensed under the MIT License.
