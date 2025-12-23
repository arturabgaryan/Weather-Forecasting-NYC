# Weather Forecasting for New York City (JFK)

This repository contains a complete machine learning pipeline for predicting
weather conditions in New York City using historical data from JFK Airport.

- data preparation and validation
- feature engineering for time-series data
- model training (regression + classification)
- model evaluation
- unified inference pipeline
- reproducibility via saved artifacts

---

##  Data(data)
### `regression_data.csv`
- features selected for regression

### `classification_data.csv`
- features selected for classification

### `data_with_month&daily_avg.csv`
- data with monthly and daily average

##  Notebooks
### `preprocessing.ipynb`
- initial data inspection
- missing value analysis
- validation of raw measurements
- application of reusable preprocessing functions

### `analysis.ipynb`
Exploratory data analysis:
- data distributions
- seasonality and trends
- missing values and data quality
- motivation for feature selection and modeling decisions

### `training_regression.ipynb`
- prepares data for temperature prediction
- trains baseline and XGBoost regression models
- evaluates performance (MAE, RMSE, R²)
- saves trained model and metadata

### `training_classification.ipynb`
- constructs weather type labels
- trains an XGBoost classifier
- evaluates accuracy and macro-averaged metrics
- analyzes class imbalance and confusion matrix
- saves model, label encoder, and configuration

### `report.ipynb`
- summary of modeling results
- comparison of baseline and final models
- interpretation of errors and limitations
- final conclusions

---

##  Preprocessing (`preprocess/`)

### `data_prep.py`
- data cleaning and validation
- missing value handling
- column filtering and sanity checks
- ensures chronological ordering

### `feature_engineering.py`
- creation of time-series features:
  - lag features
  - rolling statistics (shifted to prevent leakage)
  - seasonal features (`month`, `dayofyear`)
- shared feature logic for both models

### `labels.py`
- construction of the `weather_type` target
- label definitions based on meteorological signals
- avoids information leakage by excluding explicit weather flags

---

##  Models (`models/`)

### `regression.py`
- training logic for temperature prediction
- XGBoost regressor implementation
- clean API for model training

### `classification.py`
- training logic for weather type classification
- XGBoost classifier
- optional hyperparameter tuning support

### `metrics.py`
- centralized evaluation utilities
- regression metrics (MAE, RMSE, R²)
- classification metrics (accuracy, precision, recall, F1, confusion matrix)

---

##  Inference (`models/inference/`)

### `predict.py`
Unified inference interface:
- loads trained regression and classification models
- loads label encoder and feature configurations
- predicts:
  - maximum temperature (`predicted_TMAX`)
  - weather type (`weather_type`)
  - optional class probabilities

This module represents a real-world deployment entry point.

---

##  Model Artifacts (`models/artifacts/`)

- `models/` — serialized trained models (`.joblib`)
- `encoders/` — label encoders for classification
- `metrics/` — stored evaluation results
- `metadata/` — feature lists and model configuration files
- `parameters/` — parameters of regression model

Artifacts are stored alongside the model code to ensure reproducibility
and self-contained inference.

> [!IMPORTANT]
>
> Model **XGBoost** is required.  
>  to install:
>
> ```bash
>  pip install xgboost
>  ```
