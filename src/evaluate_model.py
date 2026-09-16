"""Evaluate the saved model predictions using standard regression metrics."""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import PREDICTIONS_PATH


def evaluate_predictions(predictions_path=PREDICTIONS_PATH):
    """Return R2, MAE and RMSE for stored test predictions."""
    data = pd.read_csv(predictions_path)
    actual = data["Actual_Optimal_Price"]
    predicted = data["Predicted_Optimal_Price"]
    return {
        "R2": r2_score(actual, predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
    }


if __name__ == "__main__":
    metrics = evaluate_predictions()
    print(f"R² Score: {metrics['R2']:.4f}")
    print(f"Mean Absolute Error: ${metrics['MAE']:.2f}")
    print(f"Root Mean Squared Error: ${metrics['RMSE']:.2f}")
