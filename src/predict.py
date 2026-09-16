import joblib
import pandas as pd
from config import MODEL_PATH


def predict_optimal_price(booking):
    """Predict an optimal price for one hotel booking dictionary."""
    model = joblib.load(MODEL_PATH)
    booking_df = pd.DataFrame([booking])
    return float(model.predict(booking_df)[0])


if __name__ == "__main__":
    booking = {
        "Base_Price": 150,
        "Season": "Summer",
        "Day_Type": "Weekend",
        "Local_Event": "Yes",
        "Room_Type": "Suite",
        "Competitor_Demand": "High",
    }
    predicted_price = predict_optimal_price(booking)
    print(f"Predicted Optimal Price: ${predicted_price:.2f}")
