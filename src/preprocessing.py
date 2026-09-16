import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from config import FEATURE_COLUMNS, TARGET_COLUMN
categorical_cols = [
    "Season", "Day_Type", "Local_Event", "Room_Type", "Competitor_Demand"
]
numeric_cols = ["Base_Price"]
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )
def split_features_target(data: pd.DataFrame):
    return data[FEATURE_COLUMNS], data[TARGET_COLUMN]
