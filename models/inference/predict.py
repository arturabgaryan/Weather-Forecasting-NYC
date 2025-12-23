import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

def load_regression_artifacts():
    model = joblib.load(
        ARTIFACTS_DIR / "models" / "tmax_xgboost.joblib"
    )

    with open(ARTIFACTS_DIR / "metadata" / "xgboost_info.json") as f:
        config = json.load(f)

    return model, config


def load_classification_artifacts():
    model = joblib.load(
        ARTIFACTS_DIR / "models" / "weather_classifier_xgb.joblib"
    )

    label_encoder = joblib.load(
        ARTIFACTS_DIR / "encoders" / "weather_label_encoder.joblib"
    )

    with open(ARTIFACTS_DIR / "metadata" / "classifier_info.json") as f:
        config = json.load(f)

    return model, label_encoder, config


def predict_weather_and_temperature(
    df: pd.DataFrame,
    return_weather_proba: bool = False
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with engineered features.
    return_weather_proba : bool, default=False
        Whether to include weather class probabilities.

    """
    df = df.copy()

    reg_model, reg_config = load_regression_artifacts()
    clf_model, label_encoder, clf_config = load_classification_artifacts()

    reg_features = reg_config["features"]
    missing_reg = set(reg_features) - set(df.columns)
    if missing_reg:
        raise ValueError(f"Missing regression features: {missing_reg}")

    df["predicted_TMAX"] = reg_model.predict(df[reg_features])

    clf_features = clf_config["features"]
    missing_clf = set(clf_features) - set(df.columns)
    if missing_clf:
        raise ValueError(f"Missing classification features: {missing_clf}")

    weather_encoded = clf_model.predict(df[clf_features])
    df["weather_type"] = label_encoder.inverse_transform(weather_encoded)

    if return_weather_proba:
        proba = clf_model.predict_proba(df[clf_features])
        proba_df = pd.DataFrame(
            proba,
            columns=label_encoder.classes_,
            index=df.index
        )
        df = pd.concat([df, proba_df], axis=1)

    return df
