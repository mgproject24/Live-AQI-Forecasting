import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import LOCATIONS, LAST_30_PATH, FORECAST_PATH, aqi_category, weather_icon

BAND_HEX = {
    "green": "#3FB950", "amber": "#D4A72C", "coral": "#E8833A",
    "red": "#E5484D", "pink": "#D6409F", "gray": "#8B949E",
}

st.set_page_config(page_title="Delhi AQI - live forecast", layout="wide", page_icon="\U0001F32B\ufe0f")

# ---------------------------------------------------------------------------
# Dark, MSN-weather-style theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: #0B1220; color: #E6EDF3; }
    section[data-testid="stSidebar"] { background: #0B1220; }
    div[data-testid="stMetric"] {
        background: #131C31; border-radius: 10px; padding: 10px 14px;
        border: 1px solid #1F2A44;
    }
    div[data-testid="stMetricLabel"] { color: #8B95A7; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #8B95A7; border-radius: 20px;
        padding: 6px 18px; margin-right: 4px;
    }
    .stTabs [aria-selected="true"] {
        background: #F5C842 !important; color: #0B1220 !important; font-weight: 600;
    }
    .forecast-card {
        background: #131C31; border: 1px solid #1F2A44; border-radius: 12px;
        padding: 14px 6px; text-align: center;
    }
    .current-card {
        background: linear-gradient(135deg, #14213D 0%, #0B1220 100%);
        border: 1px solid #1F2A44; border-radius: 14px; padding: 22px 26px;
    }
    .aqi-pill {
        padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;
        display: inline-block;
    }
    .loc-row {
        background: #131C31; border: 1px solid #1F2A44; border-radius: 10px;
        padding: 10px 16px; margin-bottom: 6px; display: flex;
        justify-content: space-between; align-items: center;
    }
    hr { border-color: #1F2A44 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=1800)
def load_data():
    last_30 = pd.read_parquet(LAST_30_PATH) if LAST_30_PATH.exists() else pd.DataFrame()
    forecast = pd.read_parquet(FORECAST_PATH) if FORECAST_PATH.exists() else pd.DataFrame()
    return last_30, forecast


last_30, forecast = load_data()

st.markdown("### \U0001F32B\ufe0f Delhi AQI \u2014 live forecast")

if last_30.empty:
    st.info(
        "No data yet. Run `python src/ingest.py --past-days 30` to backfill "
        "history, then `python src/train.py` and `python src/forecast.py`."
    )
    st.stop()

history_days = last_30["timestamp"].max() - last_30["timestamp"].min()
if history_days.days < 45:
    st.warning(
        f"Only ~{history_days.days} days of history collected so far. The "
        "7-day forecast (especially the later days) will be unreliable until "
        "the daily ingestion job has built up a couple of months of data "
        "across a full range of conditions. Treat forecasts as a "
        "work-in-progress, not a trustworthy prediction yet."
    )

col_a, col_b = st.columns([3, 1])
with col_a:
    location = st.selectbox("Location", list(LOCATIONS.keys()), label_visibility="collapsed")
with col_b:
    st.button("\U0001F504 Refresh data", width='stretch')

loc_hist = last_30[last_30["location"] == location].sort_values("timestamp")
loc_fc = forecast[forecast["location"] == location].sort_values("target_time") if not forecast.empty else pd.DataFrame()

if loc_hist.empty:
    st.warning("No history for this location yet.")
    st.stop()

latest = loc_hist.iloc[-1]
label, color_key = aqi_category(latest["aqi_index"])
hex_color = BAND_HEX.get(color_key, "#8B949E")
icon, condition = weather_icon(latest.get("weather_code"))

# ---------------------------------------------------------------------------
# Current conditions card
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="current-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px">
    <div style="display:flex;align-items:center;gap:18px">
      <div style="font-size:52px;line-height:1">{icon}</div>
      <div>
        <div style="font-size:40px;font-weight:600;line-height:1">{latest['temp_c']:.0f}\u00b0c</div>
        <div style="color:#8B95A7;font-size:14px">{condition} \u00b7 {location}</div>
      </div>
    </div>
    <div style="text-align:right">
      <span class="aqi-pill" style="background:{hex_color}22;color:{hex_color}">
        AQI {latest['aqi_index']:.0f} \u00b7 {label}
      </span>
      <div style="color:#8B95A7;font-size:12px;margin-top:6px">Updated {latest['timestamp'].strftime('%d %b, %H:%M')}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Humidity", f"{latest['humidity']:.0f} %")
m2.metric("Wind", f"{latest['windspeed_kph']:.1f} km/h")
m3.metric("Pressure", f"{latest['pressure_mb']:.0f} mb")
m4.metric("PM2.5", f"{latest.get('pm2_5', float('nan')):.0f}" if pd.notna(latest.get("pm2_5")) else "n/a")

st.markdown("---")

# ---------------------------------------------------------------------------
# 7-day forecast cards
# ---------------------------------------------------------------------------
if not loc_fc.empty:
    st.markdown("#### 7-day forecast")
    cols = st.columns(len(loc_fc))
    for c, (_, row) in zip(cols, loc_fc.iterrows()):
        f_label, f_color_key = aqi_category(row["predicted_aqi"])
        f_hex = BAND_HEX.get(f_color_key, "#8B949E")
        f_icon, _ = weather_icon(row.get("weather_code"))
        temp_txt = f"{row['temp_c']:.0f}\u00b0" if pd.notna(row.get("temp_c")) else ""
        with c:
            st.markdown(f"""
            <div class="forecast-card">
              <div style="font-size:12px;color:#8B95A7">{row['target_time'].strftime('%a')}</div>
              <div style="font-size:11px;color:#5B6478;margin-bottom:4px">{row['target_time'].strftime('%d %b')}</div>
              <div style="font-size:26px;margin:4px 0">{f_icon}</div>
              <div style="font-size:13px;color:#C9D1D9">{temp_txt}</div>
              <div style="font-size:16px;font-weight:700;color:{f_hex};margin-top:4px">{row['predicted_aqi']:.0f}</div>
              <div style="font-size:10px;color:{f_hex}">{f_label}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No forecast yet. Run `python src/train.py` then `python src/forecast.py`.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["AQI trend", "Weather"])

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.1),
)

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loc_hist["timestamp"], y=loc_hist["aqi_index"],
        name="Historical AQI (last 30 days)", mode="lines",
        line=dict(color="#5FA8FF", width=2),
    ))
    if not loc_fc.empty:
        fig.add_trace(go.Scatter(
            x=loc_fc["target_time"], y=loc_fc["predicted_aqi"],
            name="7-day forecast", mode="lines+markers",
            line=dict(dash="dash", color="#F5C842"),
        ))
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, width='stretch')

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=loc_hist["timestamp"], y=loc_hist["temp_c"],
                               name="Temp (C)", line=dict(color="#E8833A")))
    fig2.add_trace(go.Scatter(x=loc_hist["timestamp"], y=loc_hist["humidity"],
                               name="Humidity (%)", yaxis="y2", line=dict(color="#5FA8FF")))
    fig2.update_layout(yaxis2=dict(overlaying="y", side="right"), **PLOT_LAYOUT)
    st.plotly_chart(fig2, width='stretch')

st.markdown("---")

# ---------------------------------------------------------------------------
# All locations overview
# ---------------------------------------------------------------------------
st.markdown("#### All locations \u2014 current AQI")
for loc in LOCATIONS:
    d = last_30[last_30["location"] == loc]
    if d.empty:
        continue
    latest_row = d.sort_values("timestamp").iloc[-1]
    lbl, ck = aqi_category(latest_row["aqi_index"])
    hx = BAND_HEX.get(ck, "#8B949E")
    ic, _ = weather_icon(latest_row.get("weather_code"))
    st.markdown(f"""
    <div class="loc-row">
      <div style="display:flex;align-items:center;gap:10px">
        <span style="font-size:20px">{ic}</span>
        <span style="font-weight:500">{loc}</span>
      </div>
      <span class="aqi-pill" style="background:{hx}22;color:{hx}">{latest_row['aqi_index']:.0f} \u00b7 {lbl}</span>
    </div>
    """, unsafe_allow_html=True)
