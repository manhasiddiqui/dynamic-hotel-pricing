import joblib
import pandas as pd

MODEL_PATH = "../model/dynamic_hotel_pricing_random_forest.joblib"

model = joblib.load(MODEL_PATH)

booking = pd.DataFrame([{
    "Base_Price": 150,
    "Season": "Summer",
    "Day_Type": "Weekend",
    "Local_Event": "Yes",
    "Room_Type": "Suite",
    "Competitor_Demand": "High"
}])

predicted_price = model.predict(booking)[0]
print(f"Predicted Optimal Price: ${predicted_price:.2f}")
