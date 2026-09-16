import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocessing import create_preprocessor

DATA_PATH = "../data/hotel_bookings.csv"
MODEL_PATH = "../model/dynamic_hotel_pricing_random_forest.joblib"

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=["Optimal_Price"])
y = data["Optimal_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"R² Score: {r2_score(y_test, predictions):.4f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test, predictions):.2f}")

joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
