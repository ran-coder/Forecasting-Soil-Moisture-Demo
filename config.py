"""
Shared configuration for the AgroSense forecasting dashboard.
Edit values here if your column names, thresholds, or model settings change.
"""

# ---- Data ----
DATA_PATH = "indoor_data.csv"
TIMESTAMP_COL = "timestamp"
TEMP_COL = "temperature"
HUMIDITY_COL = "humidity"
SOIL_COL = "soil_value"          # 0-100 scale, higher = wetter

# Resample raw ~10-min readings into clean hourly steps before modeling.
RESAMPLE_RULE = "1h"

# ---- Soil status thresholds (derived from your dataset's labeled ranges) ----
DRY_THRESHOLD = 30.0     # soil_value <= this => DRY
WET_THRESHOLD = 70.0     # soil_value >= this => WET
                          # in between => OPTIMAL

# ---- Forecast horizons (in hours) ----
FORECAST_HORIZONS = [4, 6, 8]
MAX_HORIZON = max(FORECAST_HORIZONS)

# Alert rule: warn if CURRENT hour OR any of the next ALERT_LOOKAHEAD_HOURS
# hours are predicted/observed to be DRY.
ALERT_LOOKAHEAD_HOURS = 4

# ---- Model ----
MODEL_PATH = "sarimax_model.pkl"
SARIMAX_ORDER = (2, 1, 1)            # (p, d, q) -- no seasonal component;
SARIMAX_SEASONAL_ORDER = (0, 0, 0, 0) # dataset (~3-4 days) is too short for
                                       # a reliable 24h seasonal estimate.

# Minimum history (in hours) required before lag/rolling features are valid
MIN_LAG_HOURS = 6
