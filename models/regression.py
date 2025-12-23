"""
Regression models for predicting daily maximum temperature (TMAX).

Supported models:
- Linear Regression 
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor 
- CatBoost Regressor 
"""

from typing import Any

import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


def train_temperature_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
    random_state: int = 42
) -> Any:
    """
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Target variable (TMAX).
    model_type : str
        Model type. One of:
        - "linear"
        - "xgboost"
    random_state : int
        Random seed.

    """

    if model_type == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)

    else:
        raise ValueError(
            "Unsupported model_type. "
            "Choose from: linear, xgboost"
        )


    return model
