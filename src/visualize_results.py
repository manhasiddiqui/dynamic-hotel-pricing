"""Create the actual-vs-predicted price visualization."""
import matplotlib.pyplot as plt
import pandas as pd
from config import PLOT_PATH, PREDICTIONS_PATH


def create_actual_vs_predicted_plot(predictions_path=PREDICTIONS_PATH, output_path=PLOT_PATH):
    """Save a scatter plot comparing actual and predicted prices."""
    data = pd.read_csv(predictions_path)
    actual = data["Actual_Optimal_Price"]
    predicted = data["Predicted_Optimal_Price"]

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.7)
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual Optimal Price ($)")
    plt.ylabel("Predicted Optimal Price ($)")
    plt.title("Actual vs Predicted Hotel Prices")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    create_actual_vs_predicted_plot()
