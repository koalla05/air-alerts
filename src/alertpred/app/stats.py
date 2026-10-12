"""Statistics tab: basic descriptive dynamics of the alert history.

Deliberately kept small - a time-range + region picker, four summary
numbers, and at most three charts (timeline, duration distribution, and a
national breakdown shown only when viewing all of Ukraine) - rather than a
kitchen-sink dashboard.
"""
import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Matches the rest of the app's dark theme (app.py / map.py / components.py)
BG = "#0e1117"
GRID = "#334155"
MUTED = "#94a3b8"
INK = "#e2e8f0"
ACCENT = "#38bdf8"


def _time_range_options(df: pd.DataFrame) -> "dict[str, Optional[Tuple[pd.Timestamp, pd.Timestamp]]]":
    now = pd.Timestamp.now(tz="UTC")
    years = sorted(df["started_at"].dt.year.unique(), reverse=True)

    options: Dict[str, Optional[Tuple[pd.Timestamp, pd.Timestamp]]] = {
        "Last Day": (now - pd.Timedelta(days=1), now),
        "Last Week": (now - pd.Timedelta(days=7), now),
        "Last Month": (now - pd.Timedelta(days=30), now),
    }
    for y in years:
        options[f"Year {y}"] = (pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC"))
    options["All Time"] = (df["started_at"].min(), now)
    options["Custom Range"] = None
    return options


def _bucket_freq(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Pick a bucket width that keeps the timeline chart to a reasonable bar count."""
    days = (end - start).total_seconds() / 86400.0
    if days <= 3:
        return "h"
    if days <= 60:
        return "D"
    if days <= 450:
        return "W"
    return "ME"


def _empty_dark_layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(color=INK),
        showlegend=False,
    )
    return fig


def render_stats_tab(df_alerts: pd.DataFrame, region_mapping: Dict[str, Dict[str, str]]) -> None:
    st.subheader("Alert Dynamics")

    if df_alerts.empty:
        st.info("No processed alert data found yet - run `python -m alertpred.train` first.")
        return

    id_to_en = {meta["id"]: meta["en"] for meta in region_mapping.values()}
    available_ids = set(df_alerts["region_id"].unique())
    region_options: Dict[str, Optional[str]] = {"🇺🇦 All of Ukraine": None}
    for region_id, en in sorted(id_to_en.items(), key=lambda kv: kv[1]):
        if region_id in available_ids:
            region_options[en] = region_id

    range_opts = _time_range_options(df_alerts)

    col_range, col_region = st.columns([1, 1])
    with col_range:
        range_key = st.selectbox("Time range", options=list(range_opts.keys()), index=2)  # default: Last Month
    with col_region:
        region_label = st.selectbox("Region", options=list(region_options.keys()))
    region_id = region_options[region_label]

    if range_key == "Custom Range":
        min_d, max_d = df_alerts["started_at"].min().date(), df_alerts["started_at"].max().date()
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("From", value=max(min_d, max_d - datetime.timedelta(days=30)), min_value=min_d, max_value=max_d)
        with c2:
            end_date = st.date_input("To", value=max_d, min_value=min_d, max_value=max_d)
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    else:
        start_ts, end_ts = range_opts[range_key]

    df_scope = df_alerts if region_id is None else df_alerts[df_alerts["region_id"] == region_id]
    df_range = df_scope[(df_scope["started_at"] >= start_ts) & (df_scope["started_at"] < end_ts)]

    if df_range.empty:
        st.info("No alerts recorded for this region in the selected range.")
        return

    # Summary metrics
    total_alerts = len(df_range)
    total_hours = df_range["duration_min"].sum() / 60.0
    avg_duration = df_range["duration_min"].mean()
    span_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1 / 24.0)
    per_day = total_alerts / span_days

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Alerts", f"{total_alerts:,}")
    m2.metric("Total Alert Time", f"{total_hours:,.0f} h")
    m3.metric("Avg Duration", f"{avg_duration:,.0f} min")
    m4.metric("Avg per Day", f"{per_day:,.1f}")

    # Chart 1: alerts over time
    freq = _bucket_freq(start_ts, end_ts)
    timeline = df_range.set_index("started_at").resample(freq).size()
    fig_timeline = go.Figure(go.Bar(
        x=timeline.index, y=timeline.values,
        marker_color=ACCENT,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y} alerts<extra></extra>",
    ))
    fig_timeline.update_xaxes(color=MUTED, showgrid=False)
    fig_timeline.update_yaxes(color=MUTED, showgrid=True, gridcolor=GRID, title="Alerts")
    _empty_dark_layout(fig_timeline, f"Alerts Over Time ({region_label})")
    st.plotly_chart(fig_timeline, width="stretch")

    # Chart 2 (+3 for the national view): duration distribution, and -
    # only when looking at the whole country, where a per-region number
    # means something - the busiest regions in the selected range.
    if region_id is None:
        col_a, col_b = st.columns(2)
    else:
        col_a = st.container()
        col_b = None

    with col_a:
        clip_at = df_range["duration_min"].quantile(0.99)
        fig_hist = go.Figure(go.Histogram(
            x=df_range["duration_min"].clip(upper=clip_at),
            marker_color=ACCENT,
            nbinsx=30,
            hovertemplate="%{x:.0f} min<br>%{y} alerts<extra></extra>",
        ))
        fig_hist.update_xaxes(color=MUTED, showgrid=False, title="Minutes")
        fig_hist.update_yaxes(color=MUTED, showgrid=True, gridcolor=GRID, title="Alerts")
        _empty_dark_layout(fig_hist, "Duration Distribution")
        st.plotly_chart(fig_hist, width="stretch")

    if col_b is not None:
        with col_b:
            top = df_range.groupby("region_en").size().sort_values(ascending=True).tail(15)
            fig_top = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color=ACCENT,
                hovertemplate="%{y}<br>%{x} alerts<extra></extra>",
            ))
            fig_top.update_xaxes(color=MUTED, showgrid=True, gridcolor=GRID)
            fig_top.update_yaxes(color=MUTED, showgrid=False)
            _empty_dark_layout(fig_top, "Most Active Regions")
            st.plotly_chart(fig_top, width="stretch")
