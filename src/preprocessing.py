import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = [
    "Season", "Day_Type", "Local_Event", "Room_Type", "Competitor_Demand"
]
NUMERIC_COLS = ["Base_Price"]

def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("categorical",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLS),
            ("numeric", "passthrough", NUMERIC_COLS)
        ]
    )
