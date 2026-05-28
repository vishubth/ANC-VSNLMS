# Generated Assets

This directory stores generated DSP evaluation artifacts.

## Expected Outputs

### Waveforms

```text
assets/waveforms/
```

- noisy input waveform
- filtered waveform
- clean reference waveform

---

### Spectrograms

```text
assets/spectrograms/
```

- noisy spectrogram
- filtered spectrogram
- clean spectrogram

---

### Benchmarks

```text
assets/benchmarks/
```

- RMSE plots
- SNR comparison charts
- convergence visualizations

---

### Audio Samples

```text
assets/samples/
```

- noisy input audio
- filtered output audio
- clean reference audio

---

These assets should be generated using:

- evaluation/benchmark_runner.py
- visualization/waveform_visualizer.py
- visualization/spectrogram_visualizer.py
