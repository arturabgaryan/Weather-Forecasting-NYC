import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds basic time-based features derived from DATE.
    """
    df = df.copy()

    if "DATE" not in df.columns:
        raise ValueError("DATE column is required to add time features")

    df["month"] = df["DATE"].dt.month
    df["dayofyear"] = df["DATE"].dt.dayofyear

    return df



def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag-based features for temperature prediction.
    """
    df = df.copy()
    df = df.sort_values("DATE")

    if "TMAX" not in df.columns:
        raise ValueError("TMAX column is required to add lag features")

    df["TMAX_lag_1"] = df["TMAX"].shift(1)

    return df



def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling mean features (without data leakage).
    """
    df = df.copy()
    df = df.sort_values("DATE")

    #for temperature (regression)
    if "TMAX" in df.columns:
        df["TMAX_roll_7"] = (
            df["TMAX"]
            .shift(1)                 
            .rolling(window=7)
            .mean()
        )

    #for precipitation (classification)
    if "PRCP" in df.columns:
        df["PRCP_roll_3"] = (
            df["PRCP"]
            .shift(1)
            .rolling(window=3)
            .mean()
        )

        df["PRCP_roll_7"] = (
            df["PRCP"]
            .shift(1)
            .rolling(window=7)
            .mean()
        )

    return df


def add_climatology_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds monthly and daily climatological averages.
    """
    df = df.copy()

    if "month" not in df.columns or "dayofyear" not in df.columns:
        raise ValueError("month and dayofyear must exist before adding climatology")
    
    if "TMAX" in df.columns:
        monthly_avg = df.groupby("month")["TMAX"].transform("mean")
        df["monthly_avg_TMAX"] = monthly_avg

    if "TMAX" in df.columns:
        daily_avg = df.groupby("dayofyear")["TMAX"].transform("mean")
        df["daily_avg_TMAX"] = daily_avg

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline.
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_climatology_features(df)

    return df
