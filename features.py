"""
Feature engineering for the SARIMAX soil-moisture forecaster.

Design rule: every feature for row t is built ONLY from data available at
t-1 or earlier (lags, rolling stats of lags, rate-of-change of lags).
This means the exact same function used to build training features can be
reused step-by-step to recursively forecast forward, since we never need
"future" ground truth to compute a feature.
"""

import numpy as np
import pandas as pd
import config


def compute_vpd(temp_c: pd.Series, rh_percent: pd.Series) -> pd.Series:
    """
    Vapour Pressure Deficit (kPa) using the Tetens saturation vapor
    pressure equation. Higher VPD = drier air = plants/soil lose more
    moisture.
    """
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))  # saturation vp (kPa)
    ea = es * (rh_percent / 100.0)                              # actual vp (kPa)
    return es - ea


def load_raw(path: str = config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL])
    df = df.sort_values(config.TIMESTAMP_COL).set_index(config.TIMESTAMP_COL)
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw ~10-min sensor readings into clean hourly means."""
    numeric_cols = [config.TEMP_COL, config.HUMIDITY_COL, config.SOIL_COL]
    hourly = df[numeric_cols].resample(config.RESAMPLE_RULE).mean()
    hourly = hourly.interpolate(limit_direction="both")
    return hourly


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature set on an hourly-indexed dataframe that has
    at least [TEMP_COL, HUMIDITY_COL, SOIL_COL].

    Returns a dataframe with target column 'y' (soil_value at time t) and
    feature columns, with NaN rows (insufficient history) dropped.
    """
    df = hourly.copy()
    df["vpd"] = compute_vpd(df[config.TEMP_COL], df[config.HUMIDITY_COL])

    # --- lag features (soil) ---
    df["soil_lag1"] = df[config.SOIL_COL].shift(1)
    df["soil_lag2"] = df[config.SOIL_COL].shift(2)
    df["soil_lag3"] = df[config.SOIL_COL].shift(3)

    # --- rolling means (computed on lagged series so they only use the past) ---
    df["soil_roll_mean3"] = df["soil_lag1"].rolling(window=3).mean()
    df["soil_roll_mean6"] = df["soil_lag1"].rolling(window=6).mean()

    # --- rate of change per hour ---
    df["soil_roc1"] = df["soil_lag1"] - df["soil_lag2"]

    # --- lagged exogenous sensor readings (avoid needing future temp/humidity) ---
    df["temp_lag1"] = df[config.TEMP_COL].shift(1)
    df["humidity_lag1"] = df[config.HUMIDITY_COL].shift(1)
    df["vpd_lag1"] = df["vpd"].shift(1)

    df["y"] = df[config.SOIL_COL]

    feature_cols = [
        "soil_lag1", "soil_lag2", "soil_lag3",
        "soil_roll_mean3", "soil_roll_mean6", "soil_roc1",
        "temp_lag1", "humidity_lag1", "vpd_lag1",
    ]

    model_df = df[["y"] + feature_cols].dropna()
    return model_df, feature_cols


def status_from_value(value: float) -> str:
    if value <= config.DRY_THRESHOLD:
        return "DRY"
    elif value >= config.WET_THRESHOLD:
        return "WET"
    return "OPTIMAL"
