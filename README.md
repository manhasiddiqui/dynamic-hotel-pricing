# Dynamic Hotel Pricing Using Machine Learning

## Overview
This project develops a machine-learning-based system for predicting an optimal hotel room price from booking and market-related factors.

The model uses:
- Base price
- Season
- Day type
- Local event availability
- Room type
- Competitor demand

A Random Forest Regressor is trained to predict `Optimal_Price`.

## Features
- Synthetic dataset containing 1,000 hotel bookings
- One-Hot Encoding for categorical variables
- 80/20 train-test split
- Random Forest Regression
- R² and Mean Absolute Error evaluation
- Price prediction for a new booking scenario
- Actual vs. predicted price visualization

## Technologies
- Python
- pandas
- NumPy
- scikit-learn
- joblib
- Matplotlib

## Model Results
- R² Score: 0.9687
- Mean Absolute Error: $8.58
- Training samples: 800
- Testing samples: 200

## Example Prediction
For:
- Base Price: $150
- Season: Summer
- Day Type: Weekend
- Local Event: Yes
- Room Type: Suite
- Competitor Demand: High

Predicted Optimal Price: **$250.75**

## How to Run

1. Install Python.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
cd src
python train_model.py
```

4. Run the example prediction:

```bash
python predict.py
```

## Project Structure

```text
dynamic-hotel-pricing/
├── data/
│   └── hotel_bookings.csv
├── model/
│   └── dynamic_hotel_pricing_random_forest.joblib
├── results/
│   ├── actual_vs_predicted_hotel_prices.png
│   └── test_predictions.csv
├── src/
│   ├── config.py
│   ├── evaluate_model.py
│   ├── visualize_results.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── README.md
├── requirements.txt
└── statement.md
```

## Dataset Note
The dataset used in this project is synthetic and was generated specifically for demonstrating the ML workflow. Therefore, the reported performance reflects how well the model learned the patterns in this synthetic dataset and should not be interpreted as real-world hotel-market performance.

### Project modules
- `config.py` — shared paths and model settings.
- `preprocessing.py` — feature selection and one-hot encoding.
- `train_model.py` — training, testing, and model saving.
- `predict.py` — prediction for new hotel inputs.
- `evaluate_model.py` — R², MAE, and RMSE evaluation.
- `visualize_results.py` — actual-vs-predicted visualization.
