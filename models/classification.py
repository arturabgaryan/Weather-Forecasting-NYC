from typing import Literal

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def train_weather_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: Literal["xgboost"] = "xgboost",
    random_state: int = 42,
    tune_hyperparams: bool = False
):
    """
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Target labels (weather_type).
    model_type : str, default="xgboost"
        Type of model to train.
    random_state : int
        Random seed.
    """

    if model_type == "xgboost":

       base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_train)),
        random_state=random_state,
        verbosity=1,
    )

    if not tune_hyperparams:
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(np.unique(y_train)),
            random_state=random_state,
            verbosity=1,
        )

        model.fit(X_train, y_train)
        return model

    # ----------------------------
    # Hyperparameter tuning (light)
    # ----------------------------
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=random_state,
    )

    search.fit(X_train, y_train)

    return search.best_estimator_