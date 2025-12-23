import pandas as pd

def classify_weather(row: pd.Series) -> str:
    """
    Classifies weather type based on NOAA weather flags and measurements.
    """

    if row["WT03"] == 1 or row["WT11"] == 1:
        return "Stormy"

    elif row["SNOW"] > 0 or row["WT07"] == 1:
        return "Snowy"

    elif row["PRCP"] > 0 or row["WT06"] == 1:
        return "Rainy"

    elif row["WT01"] == 1 or row["WT02"] == 1:
        return "Foggy"

    else:
        return "Sunny"
