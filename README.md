# Dynamic Hotel Pricing Using Machine Learning

## Overview
This project builds a machine learning system that predicts the ideal price for a hotel room based on booking and market conditions.

The model takes into account:
- Base price
- Season
- Day type
- Whether a local event is happening
- Room type
- Competitor demand

A Random Forest Regressor is trained to predict the `Optimal_Price`.

## Features
- A synthetic dataset of 1,000 hotel bookings
- One-Hot Encoding for categorical variables
- An 80/20 train-test split
- Random Forest Regression
- Evaluation using R² and Mean Absolute Error
- Price prediction for a new booking scenario
- A visualization comparing actual vs. predicted prices
  
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
## Code Screenshots 
<img width="1053" height="590" alt="image" src="https://github.com/user-attachments/assets/cac2e639-e2d5-4b54-b667-8c18b42de820" />
<img width="1056" height="591" alt="image" src="https://github.com/user-attachments/assets/f2e0bb44-955a-4f56-9b4c-eecea4ceb31e" />

## Dataset Note
The dataset used in this project is synthetic. it was generated specifically to demonstrate the machine learning workflow. Because of this, the performance numbers reported here only show how well the model learned the patterns within this synthetic data, and shouldn't be taken as an indication of how it would perform in a real hotel market.
### Project modules
- `config.py` — shared paths and model settings.
- `preprocessing.py` — feature selection and one-hot encoding.
- `train_model.py` — training, testing, and model saving.
- `predict.py` — prediction for new hotel inputs.
- `evaluate_model.py` — R², MAE, and RMSE evaluation.
- `visualize_results.py` — actual-vs-predicted visualization.
