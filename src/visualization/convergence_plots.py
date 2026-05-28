import matplotlib.pyplot as plt


class ConvergenceVisualizer:
    """
    Visualization utilities for adaptive filter convergence analysis.
    """

    @staticmethod
    def plot_rmse_history(rmse_history: list[float]) -> None:
        plt.figure(figsize=(10, 5))

        plt.plot(rmse_history)

        plt.title("Adaptive Filter RMSE Convergence")
        plt.xlabel("Training Batch")
        plt.ylabel("RMSE")

        plt.grid(True)

        plt.show()
