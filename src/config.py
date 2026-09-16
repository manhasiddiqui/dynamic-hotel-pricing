"""Project configuration and shared constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "dynamic_hotel_pricing_random_forest.joblib"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "test_predictions.csv"
PLOT_PATH = PROJECT_ROOT / "results" / "actual_vs_predicted_hotel_prices.png"

TARGET_COLUMN = "Optimal_Price"
FEATURE_COLUMNS = [
    "Base_Price",
    "Season",
    "Day_Type",
    "Local_Event",
    "Room_Type",
    "Competitor_Demand",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20
