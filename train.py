"""
Train the SARIMAX soil-moisture forecasting model on historical data only.

Run:
    python train.py
"""

import pickle
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

import config
import features

warnings.filterwarnings("ignore")


def main():
    print(f"Loading raw data from {config.DATA_PATH} ...")
    raw = features.load_raw()

    print("Resampling to hourly means ...")
    hourly = features.resample_hourly(raw)
    print(f"  -> {len(hourly)} hourly rows "
          f"({hourly.index.min()} to {hourly.index.max()})")

    print("Building lag / rolling / VPD features ...")
    model_df, feature_cols = features.build_features(hourly)
    print(f"  -> {len(model_df)} usable rows after dropping warm-up NaNs")

    if len(model_df) < 20:
        raise ValueError(
            "Not enough hourly data to train reliably. Need at least ~20 "
            "valid hourly rows after feature lagging."
        )

    y = model_df["y"]
    X = model_df[feature_cols]

    print(f"Fitting SARIMAX{config.SARIMAX_ORDER} "
          f"seasonal{config.SARIMAX_SEASONAL_ORDER} with exog={feature_cols} ...")
    model = SARIMAX(
        y,
        exog=X,
        order=config.SARIMAX_ORDER,
        seasonal_order=config.SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    print(fitted.summary())

    # Save everything forecast.py / app.py need to recursively roll forward:
    # the fitted model, the feature columns order, and the tail of the
    # hourly raw series (to seed lag/rolling computations going forward).
    bundle = {
        "fitted_model": fitted,
        "feature_cols": feature_cols,
        "hourly_history": hourly,   # full hourly TEMP/HUMIDITY/SOIL history
    }
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved trained model bundle to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
