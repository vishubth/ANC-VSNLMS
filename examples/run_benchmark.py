import numpy as np

from src.evaluation.benchmark_runner import BenchmarkRunner


clean_signal = np.random.randn(1000)
noisy_signal = clean_signal + 0.5 * np.random.randn(1000)
filtered_signal = clean_signal + 0.1 * np.random.randn(1000)


results = BenchmarkRunner.evaluate(
    clean_signal,
    noisy_signal,
    filtered_signal,
)


print("Benchmark Results")
print(results)
