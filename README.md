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

##  Project Structure
Eurail/
├── data/
│ ├── raw/
│ │  └── JFK_Airport_Weather_Data.csv
│ └── processed/
│    ├── classification_data.csv
│    ├── data_with_month&daily_avg.csv
│    └── regression_data.csv
│
├── notebooks/
│ ├── analysis.ipynb
│ ├── training_regression.ipynb
│ └── training_classification.ipynb
│
├── preprocess/
│ ├── init.py
│ ├── data_prep.py
│ ├── feature_engineering.py
│ └── labels.py
│
├── models/
│ ├── regression.py
│ ├── classification.py
│ ├── metrics.py
│ ├── inference/
│ │  ├── init.py
│ │  └── predict.py
│ └── artifacts/
│    ├── models/
│    ├── encoders/
│    ├── params/
│    ├── metrics/
│    └── metadata/
│
└── README.md


---

##  Notebooks

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

Artifacts are stored alongside the model code to ensure reproducibility
and self-contained inference.
