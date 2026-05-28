# ANC-VSNLMS Architecture

## Overview

ANC-VSNLMS is a modular adaptive signal processing system designed for Active Noise Cancellation (ANC) using the Variable Step-size Normalized Least Mean Squares (VSNLMS) algorithm.

The architecture separates:

- adaptive optimization logic
- DSP filtering operations
- runtime audio processing
- evaluation and benchmarking
- experimentation workflows

This separation improves maintainability, experimentation velocity, and deployment readiness.

---

# System Components

## 1. VSNLMS Core

File:

```text
model/vsnlms.py
```

Responsibilities:

- adaptive coefficient optimization
- variable step-size adjustment
- normalization handling
- convergence stabilization
- divergence prevention

Core capabilities:

- dynamic learning-rate adaptation
- normalized updates against signal energy
- coefficient norm stabilization
- bounded adaptive behavior

---

## 2. Adaptive Filter Layer

File:

```text
model/adaptive_filter.py
```

Responsibilities:

- training orchestration
- filter application
- RMSE tracking
- filter state persistence
- inference pipeline execution

This layer acts as the orchestration engine between:

- DSP algorithms
- runtime processing
- evaluation logic

---

## 3. Real-Time Audio Runtime

File:

```text
model/live_audio_filter.py
```

Responsibilities:

- live microphone capture
- runtime preprocessing
- streaming audio filtering
- output generation
- runtime audio normalization

This component simulates deployment-oriented ANC processing pipelines.

---

# Signal Flow

```text
Microphone Input
        ↓
Signal Preprocessing
        ↓
Adaptive VSNLMS Filter
        ↓
Error Minimization
        ↓
Noise-Suppressed Output
```

---

# Adaptive Learning Workflow

```text
Input Signal
      ↓
Filter Prediction
      ↓
Residual Error Computation
      ↓
Dynamic Step-size Adjustment
      ↓
Normalized Weight Update
      ↓
Coefficient Stabilization
```

---

# Engineering Considerations

## Convergence Stability

The implementation includes:

- normalized updates
- bounded weight scaling
- adaptive learning-rate tuning

to reduce instability during changing acoustic conditions.

---

## Numerical Stability

The algorithm introduces normalization safeguards to prevent:

- exploding coefficient magnitudes
- instability during low-energy signal windows
- divergence caused by amplitude variation

---

## Real-Time Constraints

The architecture prioritizes:

- low computational overhead
- streaming compatibility
- incremental coefficient updates
- lightweight memory usage

for real-time audio applications.

---

# Potential Future Extensions

## Evaluation Layer

Potential additions:

- SNR benchmarking
- spectral analysis
- latency measurement
- convergence visualization
- spectrogram comparison

---

## Production Enhancements

Potential future directions:

- asynchronous audio streaming
- GPU acceleration
- multi-channel ANC
- embedded deployment
- edge inference optimization
- adaptive hyperparameter tuning

---

# Repository Evolution

The repository originated as a client-focused experimental ANC implementation and evolved into a modular DSP experimentation platform focused on:

- adaptive signal processing
- real-time filtering
- optimization experimentation
- deployment-oriented audio pipelines
