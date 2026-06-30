"""
AgroSense Forecasting Dashboard (historical-data simulation, no live sensors).

Run:
    streamlit run app.py
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import features
import forecast

st.set_page_config(page_title="AgroSense Forecast Dashboard", layout="wide")


# ---------------------------------------------------------------- caching --
@st.cache_resource
def get_bundle():
    if not os.path.exists(config.MODEL_PATH):
        return None
    return forecast.load_bundle()


@st.cache_data
def get_hourly_history():
    raw = features.load_raw()
    return features.resample_hourly(raw)


@st.cache_data
def get_forecast(_bundle, n_hours):
    return forecast.recursive_forecast(_bundle, n_hours)


hourly = get_hourly_history()
bundle = get_bundle()

if bundle is None:
    st.error(
        "No trained model found. Run `python train.py` first to create "
        f"`{config.MODEL_PATH}`, then reload this page."
    )
    st.stop()

n_points = len(hourly)
min_window = config.MIN_LAG_HOURS + 1

st.title("🌱 AgroSense — Historical Forecast Dashboard")

#  sidebar 
st.sidebar.title("⚙️ Playback Controls")
st.sidebar.caption(
    "This dashboard replays PAST sensor data to simulate a live feed. "
    "No real-time sensors are used — forecasts are generated purely from "
    "historical patterns."
)

if "playhead" not in st.session_state:
    st.session_state.playhead = n_points
if "live" not in st.session_state:
    st.session_state.live = False

live = st.sidebar.checkbox("🔴 Go live (auto-advance)", value=st.session_state.live)
st.session_state.live = live

speed = st.sidebar.slider("Update interval (seconds)", 0.5, 3.0, 1.0)

# Manual slider only usable when not live (live mode drives playhead itself)
manual_playhead = st.sidebar.slider(
    "Replay up to hour #", min_window, n_points, st.session_state.playhead,
    disabled=live,
    help="Drag to rewind/scrub the simulation. Disabled while 'Go live' is on.",
)
if not live:
    st.session_state.playhead = manual_playhead

if st.sidebar.button("⏮️ Reset to start"):
    st.session_state.playhead = min_window


#  live sensor panel 
@st.fragment(run_every=speed if st.session_state.live else None)
def live_sensor_panel():
    if st.session_state.live:
        if st.session_state.playhead < n_points:
            st.session_state.playhead += 1
        else:
            st.session_state.playhead = min_window  # loop back to start

    playhead = st.session_state.playhead
    view = hourly.iloc[:playhead]
    current_row = view.iloc[-1]
    current_status = features.status_from_value(current_row[config.SOIL_COL])

    status_text = "🔴 LIVE" if st.session_state.live else "⏸ Paused"
    st.caption(
        f"{status_text}  |  Simulated time: **{view.index[-1]}**  |  "
        f"({playhead}/{n_points} hourly points)"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡️ Temperature (°C)", f"{current_row[config.TEMP_COL]:.1f}")
    c2.metric("💧 Humidity (%)", f"{current_row[config.HUMIDITY_COL]:.1f}")
    c3.metric("🪴 Soil Moisture", f"{current_row[config.SOIL_COL]:.1f}", current_status)
    vpd_now = features.compute_vpd(
        pd.Series([current_row[config.TEMP_COL]]),
        pd.Series([current_row[config.HUMIDITY_COL]]),
    ).iloc[0]
    c4.metric("🍃 VPD (kPa)", f"{vpd_now:.2f}")

    status_color = {"DRY": "🔴", "OPTIMAL": "🟢", "WET": "🔵"}[current_status]
    st.subheader(f"Soil status: {status_color} {current_status}")

    #  chart 
    st.divider()
    st.header("📊 Sensor Trends")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view.index, y=view[config.SOIL_COL],
        name="Soil Moisture", mode="lines+markers",
        line=dict(color="seagreen"),
    ))
    fig.add_hline(y=config.DRY_THRESHOLD, line_dash="dot", line_color="red",
                  annotation_text="DRY threshold")
    fig.add_hline(y=config.WET_THRESHOLD, line_dash="dot", line_color="blue",
                  annotation_text="WET threshold")
    fig.update_layout(
        height=380, xaxis_title="Time", yaxis_title="Soil Moisture (0-100)",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"soil_chart_{playhead}")

    c1, c2 = st.columns(2)
    with c1:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=view.index, y=view[config.TEMP_COL],
                                    name="Temperature", line=dict(color="firebrick")))
        fig_t.update_layout(height=280, yaxis_title="°C", xaxis_title="Time",
                             title="Temperature")
        st.plotly_chart(fig_t, use_container_width=True, key=f"temp_chart_{playhead}")
    with c2:
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=view.index, y=view[config.HUMIDITY_COL],
                                    name="Humidity", line=dict(color="steelblue")))
        fig_h.update_layout(height=280, yaxis_title="%", xaxis_title="Time",
                             title="Humidity")
        st.plotly_chart(fig_h, use_container_width=True, key=f"hum_chart_{playhead}")

    with st.expander("Raw hourly sensor history (up to current point)"):
        st.dataframe(view.tail(50))


live_sensor_panel()

#  forecast 
@st.fragment(run_every=speed if st.session_state.live else None)
def forecast_panel():
    st.divider()
    st.header("📈 Forecast")
    st.caption(
        "Forecast rolls forward from the current playback position above "
        "using the same trained model coefficients (no retraining)."
    )

    # baru add to make the forecast moves like real time 
    playhead_hourly = hourly.iloc[:st.session_state.playhead]
    sim_bundle = {
    "fitted_model": bundle["fitted_model"],   # reuse the same trained model
    "feature_cols": bundle["feature_cols"],
    "hourly_history": playhead_hourly,        # but roll forward from the playhead instead
    }

    fc = forecast.recursive_forecast(sim_bundle, config.MAX_HORIZON)
    latest_row = playhead_hourly.iloc[-1]
    latest_status = features.status_from_value(latest_row[config.SOIL_COL])

    alert = forecast.get_alert(latest_status, fc)
    if alert["alert"]:
        st.error(alert["message"])
    else:
        st.success(alert["message"])

    cols = st.columns(len(config.FORECAST_HORIZONS) + 1)
    cols[0].metric(
        "🪴 Current soil moisture",
        f"{latest_row[config.SOIL_COL]:.1f}",
        latest_status,
    )
    for col, h in zip(cols[1:], config.FORECAST_HORIZONS):
        row = fc.loc[fc["hours_ahead"] == h].iloc[0]
        delta = row["forecast_soil_value"] - latest_row[config.SOIL_COL]
        col.metric(
            f"+{h}h forecast",
            f"{row['forecast_soil_value']:.1f}",
            f"{delta:+.1f} vs latest ({row['status']})",
        )

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=playhead_hourly.tail(24).index, y=playhead_hourly[config.SOIL_COL],
        name="Soil Moisture (recent history)", mode="lines+markers",
        line=dict(color="seagreen"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc.index, y=fc["forecast_soil_value"],
        name="Soil Moisture (forecast)", mode="lines+markers",
        line=dict(color="orange", dash="dash"),
    ))
    fig_fc.add_hline(y=config.DRY_THRESHOLD, line_dash="dot", line_color="red")
    fig_fc.add_hline(y=config.WET_THRESHOLD, line_dash="dot", line_color="blue")
    fig_fc.update_layout(
        height=380, xaxis_title="Time", yaxis_title="Soil Moisture (0-100)",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_fc, use_container_width=True, key=f"forecast_chart_{st.session_state.playhead}")

    with st.expander("Raw forecast table"):
        st.dataframe(fc)
forecast_panel()