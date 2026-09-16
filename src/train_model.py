import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from config import DATA_PATH, MODEL_PATH, PREDICTIONS_PATH, RANDOM_STATE, TEST_SIZE
from preprocessing import create_preprocessor, split_features_target


def train_and_save_model():
    data = pd.read_csv(DATA_PATH)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    pd.DataFrame({
        "Actual_Optimal_Price": y_test.values,
        "Predicted_Optimal_Price": predictions,
    }).to_csv(PREDICTIONS_PATH, index=False)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"R² Score: {r2_score(y_test, predictions):.4f}")
    print(f"Mean Absolute Error: ${mean_absolute_error(y_test, predictions):.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
