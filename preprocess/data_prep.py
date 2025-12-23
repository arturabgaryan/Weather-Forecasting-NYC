import pandas as pd

def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").set_index("DATE")

    #Temperature
    for col in ["TMAX", "TMIN"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time")

    #Temperature average
    if "TAVG" in df.columns:
        df["TAVG"] = df["TAVG"].fillna((df["TMAX"] + df["TMIN"]) / 2)

    #Snow related
    for col in ["PRCP", "SNOW", "SNWD"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    #Wind
    if "AWND" in df.columns:
        df["AWND"] = df["AWND"].fillna(df["AWND"].median())

    #Weather flags
    wt_cols = [c for c in df.columns if c.startswith("WT") or c.startswith("WV")]
    if wt_cols:
        df[wt_cols] = df[wt_cols].fillna(0)


    return df.reset_index()



