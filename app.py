"""
Transformer Thermal Dashboard — VET260223
=========================================
Reads pre-computed output CSVs and presents an interactive
thermal monitoring view for the Varberg Energi group.

Structure
---------
Section A  (this file)  — IEC 60076-7 Dynamic Simulation   [Arnob]
Section B  (add below)  — Heat-Balance Calibration          [Suraj]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transformer Thermal Dashboard — VET260223",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY    = "#1B2A4A"
BLUE    = "#1D5B8C"
AMBER   = "#D4850A"
GREEN   = "#1A7A5E"
RED     = "#B22222"
GREY    = "#94A3B8"
LTGREY  = "#F1F5F9"
WHITE   = "#FFFFFF"
INDIGO  = "#4F46E5"
PURPLE  = "#7C3AED"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Remove default Streamlit top padding */
  .block-container {{ padding-top: 0rem !important; }}

  /* ── Header bar ────────────────────────────────────────────────────────── */
  .dash-header {{
    background: {NAVY};
    padding: 1.1rem 2rem 0.9rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid {AMBER};
  }}
  .dash-header-left {{ display: flex; flex-direction: column; gap: 0.15rem; }}
  .dash-title {{
    font-size: 1.45rem; font-weight: 700;
    color: {WHITE}; letter-spacing: 0.01em; margin: 0;
  }}
  .dash-subtitle {{ font-size: 0.82rem; color: #A8C0D8; margin: 0; }}
  .dash-badge {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 0.5rem;
    padding: 0.35rem 0.85rem;
    font-size: 0.78rem;
    color: #C8DCF0;
    text-align: right;
    line-height: 1.6;
  }}

  /* ── Section headings ───────────────────────────────────────────────────── */
  .section-label {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: {GREY}; margin-bottom: 0.55rem;
  }}
  .section-title {{
    font-size: 1.05rem; font-weight: 700; color: {NAVY};
    margin-bottom: 0.25rem;
  }}
  .section-desc {{
    font-size: 0.82rem; color: #64748B; margin-bottom: 0.9rem;
    line-height: 1.5;
  }}

  /* ── KPI cards ──────────────────────────────────────────────────────────── */
  .kpi-card {{
    background: {WHITE};
    border-radius: 0.75rem;
    padding: 1rem 1.1rem 0.9rem 1.1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border-top: 3px solid var(--accent);
    height: 100%;
  }}
  .kpi-label  {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
                 text-transform: uppercase; color: {GREY}; margin-bottom: 0.3rem; }}
  .kpi-value  {{ font-size: 1.9rem; font-weight: 700; color: {NAVY};
                 line-height: 1.1; margin-bottom: 0.15rem; }}
  .kpi-unit   {{ font-size: 0.8rem; font-weight: 400; color: {GREY};
                 margin-left: 0.2rem; }}
  .kpi-status {{ font-size: 0.78rem; font-weight: 600; margin-top: 0.35rem;
                 padding: 0.18rem 0.55rem; border-radius: 1rem; display: inline-block; }}
  .status-green  {{ background:#D1FAE5; color:#065F46; }}
  .status-amber  {{ background:#FEF3C7; color:#92400E; }}
  .status-red    {{ background:#FEE2E2; color:#7F1D1D; }}

  /* ── Dividers ───────────────────────────────────────────────────────────── */
  .divider {{
    border: none; border-top: 1px solid #E2E8F0;
    margin: 1.6rem 0 1.3rem 0;
  }}
  .section-divider {{
    border: none; border-top: 3px solid {AMBER};
    margin: 2rem 0 1.5rem 0; border-radius: 2px;
    width: 3rem;
  }}

  /* ── Footer ─────────────────────────────────────────────────────────────── */
  .dash-footer {{
    background: {LTGREY}; border-top: 1px solid #E2E8F0;
    padding: 0.9rem 1.5rem;
    font-size: 0.75rem; color: #64748B;
    border-radius: 0.5rem; margin-top: 2rem;
    line-height: 1.7;
  }}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

@st.cache_data
def load_dtr() -> pd.DataFrame:
    df = pd.read_csv(HERE / "output" / "dtr_results.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

@st.cache_data
def load_forecast() -> pd.DataFrame:
    df = pd.read_csv(HERE / "output" / "forecast_load.csv")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df

dtr  = load_dtr()
fcst = load_forecast()

# ── Derived summary values ─────────────────────────────────────────────────────
max_hotspot    = dtr["hotspot_temp"].max()
max_oil        = dtr["oil_temp"].max()
min_kmax       = dtr["K_max"].min()
total_lol      = dtr["LOL"].max()           # cumulative minutes at ref temp
actual_k_mean  = dtr["load_factor"].mean()  # mean load factor
hotspot_limit  = 90.0                       # °C operating limit

# Safety status helpers
def hotspot_status(val: float) -> tuple[str, str]:
    if val < 70:   return "Well within limit", "status-green"
    if val < 85:   return "Monitor closely",   "status-amber"
    return "Action required", "status-red"

def kmax_status(val: float) -> tuple[str, str]:
    if val >= 1.5: return "Ample headroom",    "status-green"
    if val >= 1.0: return "Limited headroom",  "status-amber"
    return "At limit", "status-red"

def lol_status(val: float) -> tuple[str, str]:
    if val < 1:    return "Negligible ageing",  "status-green"
    if val < 10:   return "Moderate ageing",    "status-amber"
    return "High ageing", "status-red"

hs_txt,  hs_cls  = hotspot_status(max_hotspot)
kmax_txt, kmax_cls = kmax_status(min_kmax)
lol_txt,  lol_cls  = lol_status(total_lol)

# ── Chart helpers ─────────────────────────────────────────────────────────────
def base_layout(title: str = "", yaxis_title: str = "", height: int = 340) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13, color=NAVY), x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=20, t=45, b=30),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=WHITE, bordercolor="#E2E8F0",
                        font=dict(size=11, color=NAVY)),
        xaxis=dict(
            showgrid=False, linecolor="#E2E8F0", linewidth=1,
            tickformat="%d %b\n%H:%M", tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=yaxis_title, gridcolor="#F1F5F9", gridwidth=1,
            linecolor="#E2E8F0", linewidth=1,
            tickfont=dict(size=10),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
    )

# Thin DTR to every 10th row for smoother chart rendering (still 2-min × 10 = 20-min)
dtr_thin = dtr.iloc[::5].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
forecast_start = dtr["timestamp"].min().strftime("%d %b %Y  %H:%M UTC")
forecast_end   = dtr["timestamp"].max().strftime("%d %b %Y  %H:%M UTC")

st.markdown(f"""
<div class="dash-header">
  <div class="dash-header-left">
    <p class="dash-title">⚡ Transformer Thermal Dashboard</p>
    <p class="dash-subtitle">VET260223 &nbsp;·&nbsp; 40 MVA OFAF &nbsp;·&nbsp; Varberg, Sweden
       &nbsp;·&nbsp; IEC 60076-7 Dynamic Thermal Rating</p>
  </div>
  <div class="dash-badge">
    <strong>Forecast window</strong><br>
    {forecast_start}<br>
    → {forecast_end}
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — IEC 60076-7 Dynamic Simulation  (Arnob)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-label">Section A &nbsp;·&nbsp; IEC 60076-7 Dynamic Simulation</p>',
            unsafe_allow_html=True)
st.markdown('<p class="section-title">48-Hour Thermal Forecast</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">The transformer\'s predicted temperatures and safety margins over the '
    'next 48 hours, computed by solving the IEC 60076-7 thermal differential equations on the '
    'forecasted electrical load.</p>',
    unsafe_allow_html=True,
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{AMBER}">
      <p class="kpi-label">Peak Hotspot Temperature</p>
      <p class="kpi-value">{max_hotspot:.1f}<span class="kpi-unit">°C</span></p>
      <span class="kpi-status {hs_cls}">{hs_txt}</span>
      <p style="font-size:0.72rem;color:{GREY};margin-top:0.4rem;">
        Limit: {hotspot_limit:.0f} °C
      </p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{BLUE}">
      <p class="kpi-label">Peak Oil Temperature</p>
      <p class="kpi-value">{max_oil:.1f}<span class="kpi-unit">°C</span></p>
      <span class="kpi-status {hs_cls}">{hs_txt}</span>
      <p style="font-size:0.72rem;color:{GREY};margin-top:0.4rem;">
        Limit: ~85 °C
      </p>
    </div>""", unsafe_allow_html=True)

with c3:
    headroom_pct = (min_kmax - actual_k_mean) / min_kmax * 100
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{GREEN}">
      <p class="kpi-label">Safety Headroom (K<sub>max</sub>)</p>
      <p class="kpi-value">{min_kmax:.2f}<span class="kpi-unit">p.u.</span></p>
      <span class="kpi-status {kmax_cls}">{kmax_txt}</span>
      <p style="font-size:0.72rem;color:{GREY};margin-top:0.4rem;">
        Currently using {actual_k_mean*100:.1f}% of rated capacity
      </p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{PURPLE}">
      <p class="kpi-label">Insulation Ageing (48 h)</p>
      <p class="kpi-value">{total_lol:.4f}<span class="kpi-unit">min</span></p>
      <span class="kpi-status {lol_cls}">{lol_txt}</span>
      <p style="font-size:0.72rem;color:{GREY};margin-top:0.4rem;">
        Equivalent minutes at 110 °C reference
      </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Chart A1: Temperature forecast ───────────────────────────────────────────
st.markdown(
    '<p class="section-desc" style="margin-bottom:0.3rem;">'
    '<strong>Hotspot & Oil Temperature</strong> — '
    'The hotspot is the hottest point inside the transformer windings. '
    'Both temperatures must stay well below the 90 °C operating limit.</p>',
    unsafe_allow_html=True,
)

fig_temp = go.Figure()

# Ambient temperature (shaded area at bottom)
fig_temp.add_trace(go.Scatter(
    x=dtr_thin["timestamp"], y=dtr_thin["ambient_temp"],
    name="Ambient temperature",
    line=dict(color=GREY, width=1.2, dash="dot"),
    fill="tozeroy", fillcolor="rgba(148,163,184,0.10)",
    hovertemplate="%{y:.1f} °C<extra>Ambient</extra>",
))

# Oil temperature
fig_temp.add_trace(go.Scatter(
    x=dtr_thin["timestamp"], y=dtr_thin["oil_temp"],
    name="Oil temperature",
    line=dict(color=BLUE, width=2.2),
    hovertemplate="%{y:.1f} °C<extra>Oil temp</extra>",
))

# Hotspot temperature
fig_temp.add_trace(go.Scatter(
    x=dtr_thin["timestamp"], y=dtr_thin["hotspot_temp"],
    name="Hotspot temperature",
    line=dict(color=AMBER, width=2.5),
    hovertemplate="%{y:.1f} °C<extra>Hotspot</extra>",
))

# Safety limit line
fig_temp.add_hline(
    y=hotspot_limit, line_dash="dash", line_color=RED, line_width=1.5,
    annotation_text="Safety limit (90 °C)",
    annotation_position="top right",
    annotation_font=dict(color=RED, size=10),
)

fig_temp.update_layout(**base_layout(yaxis_title="Temperature (°C)", height=340))
fig_temp.update_yaxes(range=[0, 100])
st.plotly_chart(fig_temp, use_container_width=True)

# ── Chart A2: Safety headroom ─────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc" style="margin-bottom:0.3rem;">'
    '<strong>Safety Headroom — How Much More Load Can the Transformer Handle?</strong> — '
    'The green area shows the spare loading capacity available at every moment. '
    'The dark area at the bottom is the actual load being served. '
    'The higher the green area, the safer the operating state.</p>',
    unsafe_allow_html=True,
)

fig_kmax = go.Figure()

# Available headroom (K_max - actual K) as green fill
fig_kmax.add_trace(go.Scatter(
    x=dtr_thin["timestamp"],
    y=dtr_thin["K_max"],
    name="Maximum safe load (K_max)",
    line=dict(color=GREEN, width=0),
    fill="tonexty",
    fillcolor="rgba(26,122,94,0.15)",
    hovertemplate="%{y:.2f} p.u.<extra>K_max</extra>",
    showlegend=False,
))

# Actual load factor
fig_kmax.add_trace(go.Scatter(
    x=dtr_thin["timestamp"],
    y=dtr_thin["load_factor"],
    name="Actual load (K)",
    line=dict(color=NAVY, width=2),
    fill="tozeroy",
    fillcolor=f"rgba(27,42,74,0.25)",
    hovertemplate="%{y:.3f} p.u.<extra>Actual load</extra>",
))

# K_max line on top
fig_kmax.add_trace(go.Scatter(
    x=dtr_thin["timestamp"],
    y=dtr_thin["K_max"],
    name="Maximum safe load (K_max)",
    line=dict(color=GREEN, width=2),
    hovertemplate="%{y:.2f} p.u.<extra>K_max</extra>",
))

# Rated = 1.0 reference
fig_kmax.add_hline(
    y=1.0, line_dash="dot", line_color="#64748B", line_width=1.2,
    annotation_text="Rated capacity (1.0 p.u.)",
    annotation_position="bottom right",
    annotation_font=dict(color="#64748B", size=10),
)

fig_kmax.update_layout(**base_layout(yaxis_title="Load factor (p.u.)", height=300))
fig_kmax.update_yaxes(range=[0, 2.3])
st.plotly_chart(fig_kmax, use_container_width=True)

# ── Charts A3 + A4: Load forecast & LOL ──────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

col_load, col_lol = st.columns([3, 2])

with col_load:
    st.markdown(
        '<p class="section-desc" style="margin-bottom:0.3rem;">'
        '<strong>Forecasted Electrical Load</strong> — '
        'Predicted demand on the transformer (MW), which drives the thermal simulation above. '
        'Forecast is produced by a machine-learning model trained on historical load and weather data.</p>',
        unsafe_allow_html=True,
    )
    fig_load = go.Figure()
    fig_load.add_trace(go.Bar(
        x=fcst["timestamp_utc"], y=fcst["load_mw"],
        name="Load forecast (MW)",
        marker=dict(color=INDIGO, opacity=0.8),
        hovertemplate="%{y:.2f} MW<extra>Load forecast</extra>",
    ))
    load_layout = base_layout(yaxis_title="Load (MW)", height=280)
    load_layout["bargap"] = 0.15
    fig_load.update_layout(**load_layout)
    fig_load.update_yaxes(range=[0, fcst["load_mw"].max() * 1.35])
    st.plotly_chart(fig_load, use_container_width=True)

with col_lol:
    st.markdown(
        '<p class="section-desc" style="margin-bottom:0.3rem;">'
        '<strong>Insulation Ageing Accumulation</strong> — '
        'How much insulation life is consumed over the forecast window. '
        'Measured in equivalent minutes at the 110 °C reference temperature.</p>',
        unsafe_allow_html=True,
    )
    fig_lol = go.Figure()
    fig_lol.add_trace(go.Scatter(
        x=dtr_thin["timestamp"], y=dtr_thin["LOL"],
        name="Cumulative ageing",
        line=dict(color=PURPLE, width=2.2),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.12)",
        hovertemplate="%{y:.4f} min<extra>Ageing</extra>",
    ))
    lol_layout = base_layout(yaxis_title="Equivalent minutes", height=280)
    fig_lol.update_layout(**lol_layout)
    st.plotly_chart(fig_lol, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — Heat-Balance Calibration
# ═══════════════════════════════════════════════════════════════════════════════

import yaml

@st.cache_data
def load_hb_daily() -> pd.DataFrame:
    df = pd.read_csv(HERE / "output" / "hb_forecast_daily.csv")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df

@st.cache_data
def load_iec_daily() -> pd.DataFrame:
    df = pd.read_csv(HERE / "output" / "iec_forecast_daily.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

@st.cache_data
def load_inspected_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(HERE / "output" / "inspected_values.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def _to_df(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records).copy()
        df["yymm"] = df["yymm"].astype(int)
        yy = (df["yymm"] // 100).astype(int)
        mm = (df["yymm"] % 100).astype(int)
        df["timestamp_utc"] = pd.to_datetime(
            {"year": 2000 + yy, "month": mm, "day": 15},
            utc=True,
        )
        df["t_mid"] = (df["t_min"] + df["t_max"]) / 2
        return df.sort_values("timestamp_utc").reset_index(drop=True)

    oil_df = _to_df(data.get("oil", []))
    coil_df = _to_df(data.get("coil", []))
    return oil_df, coil_df

hb_daily = load_hb_daily()
iec_daily = load_iec_daily()
oil_inspected, coil_inspected = load_inspected_values()
x_start = pd.Timestamp("2025-09-01", tz="UTC")
cal = pd.merge(
    hb_daily,
    iec_daily,
    left_on="timestamp_utc",
    right_on="timestamp",
    how="inner",
).sort_values("timestamp_utc")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    '<p class="section-label">Section B &nbsp;·&nbsp; Heat-Balance Calibration</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-title">Daily Model Comparison</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">These heat-balance forecasts are calibrated with the inspected monthly temperature values shown on the plots below.</p>',
    unsafe_allow_html=True,
)

b1, b2, b3 = st.columns(3)

with b1:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{GREEN}">
      <p class="kpi-label">HB Peak Oil Temperature</p>
      <p class="kpi-value">{cal["T_oil_hb"].max():.1f}<span class="kpi-unit">°C</span></p>
    </div>""", unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{BLUE}">
      <p class="kpi-label">IEC Peak Oil Temperature</p>
      <p class="kpi-value">{cal["T_oil_iec"].max():.1f}<span class="kpi-unit">°C</span></p>
    </div>""", unsafe_allow_html=True)

with b3:
    oil_gap = cal["T_oil_hb"] - cal["T_oil_iec"]
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{AMBER}">
      <p class="kpi-label">Mean Oil Temperature Gap</p>
      <p class="kpi-value">{oil_gap.mean():.1f}<span class="kpi-unit">°C</span></p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<p class="section-desc" style="margin-bottom:0.3rem;">'
    '<strong>Oil Temperature Comparison</strong></p>',
    unsafe_allow_html=True,
)

fig_oil_compare = go.Figure()

fig_oil_compare.add_trace(go.Scatter(
    x=cal["timestamp_utc"],
    y=cal["T_oil_iec"],
    name="IEC oil temperature",
    line=dict(color=BLUE, width=2.2, dash="dash"),
    hovertemplate="%{y:.1f} °C<extra>IEC oil</extra>",
))

fig_oil_compare.add_trace(go.Scatter(
    x=cal["timestamp_utc"],
    y=cal["hi68"],
    mode="lines",
    line=dict(width=0),
    hoverinfo="skip",
    showlegend=False,
))

fig_oil_compare.add_trace(go.Scatter(
    x=cal["timestamp_utc"],
    y=cal["lo68"],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(26,122,94,0.14)",
    name="HB 68% interval",
    hovertemplate="%{y:.1f} °C<extra>HB 68%</extra>",
))

fig_oil_compare.add_trace(go.Scatter(
    x=cal["timestamp_utc"],
    y=cal["T_oil_hb"],
    name="Heat-balance oil temperature",
    line=dict(color=GREEN, width=2.5),
    hovertemplate="%{y:.1f} °C<extra>HB oil</extra>",
))

# inspected oil values: min-max range + midpoint
fig_oil_compare.add_trace(go.Scatter(
    x=oil_inspected["timestamp_utc"],
    y=oil_inspected["t_mid"],
    mode="markers",
    name="Inspected oil values",
    marker=dict(color=RED, size=8, symbol="diamond"),
    error_y=dict(
        type="data",
        symmetric=False,
        array=oil_inspected["t_max"] - oil_inspected["t_mid"],
        arrayminus=oil_inspected["t_mid"] - oil_inspected["t_min"],
        thickness=1.5,
        width=6,
        color=RED,
    ),
    hovertemplate=(
        "Inspected oil<br>"
        "Min: %{customdata[0]:.1f} °C<br>"
        "Max: %{customdata[1]:.1f} °C<br>"
        "Mid: %{y:.1f} °C<extra></extra>"
    ),
    customdata=np.c_[oil_inspected["t_min"], oil_inspected["t_max"]],
))

fig_oil_compare.update_layout(**base_layout(yaxis_title="Oil temperature (°C)", height=340))
fig_oil_compare.update_xaxes(range=[x_start, cal["timestamp_utc"].max()])
st.plotly_chart(fig_oil_compare, use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

col_hs, col_gap = st.columns(2)

with col_hs:
    st.markdown(
        '<p class="section-desc" style="margin-bottom:0.3rem;">'
        '<strong>Hotspot Temperature Comparison</strong></p>',
        unsafe_allow_html=True,
    )

    fig_hs_compare = go.Figure()

    fig_hs_compare.add_trace(go.Scatter(
        x=cal["timestamp_utc"],
        y=cal["T_hs_iec"],
        name="IEC hotspot temperature",
        line=dict(color=PURPLE, width=2.2, dash="dash"),
        hovertemplate="%{y:.1f} °C<extra>IEC hotspot</extra>",
    ))

    fig_hs_compare.add_trace(go.Scatter(
        x=cal["timestamp_utc"],
        y=cal["hs_hi68"],
        mode="lines",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig_hs_compare.add_trace(go.Scatter(
        x=cal["timestamp_utc"],
        y=cal["hs_lo68"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(124,58,237,0.14)",
        name="HB 68% interval",
        hovertemplate="%{y:.1f} °C<extra>HB 68%</extra>",
    ))

    fig_hs_compare.add_trace(go.Scatter(
        x=cal["timestamp_utc"],
        y=cal["T_hs_hb"],
        name="Heat-balance hotspot temperature",
        line=dict(color=AMBER, width=2.5),
        hovertemplate="%{y:.1f} °C<extra>HB hotspot</extra>",
    ))

    # inspected coil values: min-max range + midpoint
    fig_hs_compare.add_trace(go.Scatter(
        x=coil_inspected["timestamp_utc"],
        y=coil_inspected["t_mid"],
        mode="markers",
        name="Inspected coil values",
        marker=dict(color=RED, size=8, symbol="diamond"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=coil_inspected["t_max"] - coil_inspected["t_mid"],
            arrayminus=coil_inspected["t_mid"] - coil_inspected["t_min"],
            thickness=1.5,
            width=6,
            color=RED,
        ),
        hovertemplate=(
            "Inspected coil<br>"
            "Min: %{customdata[0]:.1f} °C<br>"
            "Max: %{customdata[1]:.1f} °C<br>"
            "Mid: %{y:.1f} °C<extra></extra>"
        ),
        customdata=np.c_[coil_inspected["t_min"], coil_inspected["t_max"]],
    ))

    fig_hs_compare.update_layout(**base_layout(yaxis_title="Hotspot temperature (°C)", height=320))
    fig_hs_compare.update_xaxes(range=[x_start, cal["timestamp_utc"].max()])
    st.plotly_chart(fig_hs_compare, use_container_width=True)

with col_gap:
    st.markdown(
        '<p class="section-desc" style="margin-bottom:0.3rem;">'
        '<strong>Daily Oil Temperature Gap</strong></p>',
        unsafe_allow_html=True,
    )

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        x=cal["timestamp_utc"],
        y=cal["T_oil_hb"] - cal["T_oil_iec"],
        name="HB - IEC",
        marker=dict(color=RED, opacity=0.75),
        hovertemplate="%{y:.1f} °C<extra>HB - IEC</extra>",
    ))
    fig_gap.add_hline(
        y=0,
        line_dash="dot",
        line_color="#64748B",
        line_width=1.2,
    )
    gap_layout = base_layout(yaxis_title="Temperature gap (°C)", height=320)
    gap_layout["bargap"] = 0.2
    fig_gap.update_layout(**gap_layout)
    st.plotly_chart(fig_gap, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="dash-footer">
  <strong>Data & Methodology</strong> &nbsp;·&nbsp;
  Thermal simulation follows <strong>IEC 60076-7:2018</strong> using explicit Euler integration
  at 2-minute sub-steps. &nbsp;
  Load forecast produced by a <strong>LightGBM</strong> model trained on historical measurements
  from Varberg Energi (transformer VET260223) and weather reanalysis data (Open-Meteo ERA5). &nbsp;
  Nameplate parameters: 40 MVA · OFAF cooling · Δθ_or = 52 K · Δθ_hr = 26 K. &nbsp;
  Operating hotspot limit: 90 °C (conservative operator setting). &nbsp;
  Insulation ageing expressed as equivalent minutes at the IEC reference temperature of 110 °C.
<br><br>
<strong>Calibration Note</strong> &nbsp;·&nbsp;
A consistent ~36.5 °C gap is observed between modelled and inspected temperatures. Calibration assumes the inspected values correspond to this transformer and adjusts total loss, core loss, and heat capacity accordingly. The largely constant offset suggests a systematic thermal bias — potentially due to operation in a closed or semi-closed enclosure increasing local ambient temperature beyond model assumptions.
  <br><br>
  Built by <strong>Arnob</strong> (Section A) and <strong>Suraj</strong> (Section B) &nbsp;·&nbsp;
  For the Varberg Energi transformer monitoring group &nbsp;·&nbsp; April 2026
</div>
""", unsafe_allow_html=True)
