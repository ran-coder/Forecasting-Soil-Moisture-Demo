"""
Recursive multi-step forecasting on top of the trained SARIMAX model.

Because no real-time/live data is used, every step of the forecast is
generated purely from: (a) the trained model, and (b) previously
observed or previously *forecasted* values. Temperature/humidity beyond
the last real reading are persisted at their last observed value, which
is a reasonable simplifying assumption for a controlled indoor
environment and is clearly surfaced in the dashboard UI.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config
import features

warnings.filterwarnings("ignore", category=Warning, module="statsmodels")


def load_bundle(path: str = config.MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_next_feature_row(hist: pd.DataFrame) -> pd.Series:
    """
    Given a hourly history dataframe (with TEMP/HUMIDITY/SOIL columns)
    whose LAST row is time t-1, build the feature row needed to predict
    time t. Mirrors features.build_features() but for a single new step.
    """
    soil = hist[config.SOIL_COL]
    temp = hist[config.TEMP_COL]
    humidity = hist[config.HUMIDITY_COL]

    soil_lag1 = soil.iloc[-1]
    soil_lag2 = soil.iloc[-2]
    soil_lag3 = soil.iloc[-3]
    soil_roll_mean3 = soil.iloc[-3:].mean()
    soil_roll_mean6 = soil.iloc[-6:].mean()
    soil_roc1 = soil_lag1 - soil_lag2

    temp_lag1 = temp.iloc[-1]
    humidity_lag1 = humidity.iloc[-1]
    vpd_lag1 = features.compute_vpd(pd.Series([temp_lag1]),
                                     pd.Series([humidity_lag1])).iloc[0]

    return pd.Series({
        "soil_lag1": soil_lag1,
        "soil_lag2": soil_lag2,
        "soil_lag3": soil_lag3,
        "soil_roll_mean3": soil_roll_mean3,
        "soil_roll_mean6": soil_roll_mean6,
        "soil_roc1": soil_roc1,
        "temp_lag1": temp_lag1,
        "humidity_lag1": humidity_lag1,
        "vpd_lag1": vpd_lag1,
    })


def recursive_forecast(bundle: dict, n_hours: int) -> pd.DataFrame:
    """
    Roll the model forward n_hours steps, one hour at a time, feeding
    each prediction back into history so lag/rolling features stay valid.

    Each step rebuilds a fresh SARIMAX model on the full accumulated
    history (works regardless of where `hourly_history` is truncated to,
    e.g. an earlier playhead) and conditions it on that data using the
    already-fitted parameters (`.filter(params)`), rather than trying to
    `.append()`/`.apply()` onto the original fitted results object --
    which either requires the index to extend the ORIGINAL training
    index, or (with `.apply()`) silently discards all prior history.

    Returns a dataframe indexed by future timestamp with columns:
    ['forecast_soil_value', 'status', 'temp_assumed', 'humidity_assumed']
    """
    fitted = bundle["fitted_model"]
    feature_cols = bundle["feature_cols"]
    hist = bundle["hourly_history"].copy()

    last_temp = hist[config.TEMP_COL].iloc[-1]
    last_humidity = hist[config.HUMIDITY_COL].iloc[-1]
    last_ts = hist.index[-1]

    results = []

    for step in range(1, n_hours + 1):
        feat_row = _build_next_feature_row(hist)
        next_ts = last_ts + pd.Timedelta(hours=step)
        X_next = feat_row[feature_cols].to_frame().T
        X_next.index = pd.DatetimeIndex([next_ts], freq="h")

        # Rebuild the full feature/endog series from accumulated history
        # (truncated original history + any forecasts appended so far),
        # then condition the already-fitted parameters on it.
        model_df, _ = features.build_features(hist)
        y_full = model_df["y"]
        X_full = model_df[feature_cols]

        step_model = SARIMAX(
            y_full,
            exog=X_full,
            order=config.SARIMAX_ORDER,
            seasonal_order=config.SARIMAX_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        step_results = step_model.filter(fitted.params)

        pred = step_results.get_forecast(steps=1, exog=X_next)
        yhat = float(pred.predicted_mean.iloc[0])
        yhat = float(np.clip(yhat, 0, 100))  # soil_value is a 0-100 scale

        # Append the forecasted point back into history so the NEXT
        # iteration's lag/rolling features (and rebuild) include it.
        new_row = pd.DataFrame(
            {
                config.TEMP_COL: [last_temp],       # persisted assumption
                config.HUMIDITY_COL: [last_humidity],  # persisted assumption
                config.SOIL_COL: [yhat],
            },
            index=[next_ts],
        )
        hist = pd.concat([hist, new_row])

        results.append({
            "timestamp": next_ts,
            "hours_ahead": step,
            "forecast_soil_value": yhat,
            "status": features.status_from_value(yhat),
            "temp_assumed": last_temp,
            "humidity_assumed": last_humidity,
        })

    return pd.DataFrame(results).set_index("timestamp")


def get_alert(current_status: str, forecast_df: pd.DataFrame) -> dict:
    """
    Pump alert rule: trigger if the CURRENT hour OR any hour within the
    next config.ALERT_LOOKAHEAD_HOURS is DRY.
    """
    lookahead = forecast_df[
        forecast_df["hours_ahead"] <= config.ALERT_LOOKAHEAD_HOURS
    ]
    dry_now = current_status == "DRY"
    dry_soon = (lookahead["status"] == "DRY").any()

    if dry_now or dry_soon:
        if dry_now:
            reason = "Current soil reading is DRY."
        else:
            dry_hours = lookahead.loc[lookahead["status"] == "DRY", "hours_ahead"].tolist()
            reason = f"Soil is predicted to be DRY in the next {min(dry_hours)}h."
        return {"alert": True, "message": f"⚠️ PUMP WATER — {reason}"}

    return {"alert": False, "message": "✅ Soil moisture is fine — no pumping needed."}