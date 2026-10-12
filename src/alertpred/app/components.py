from typing import Dict, Any, List
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import datetime


def render_probability_bars(probabilities: Dict[str, float], is_active: bool):
    """Compact probability stat chips, three across, matching the main
    prediction card's visual style instead of stacked full-width rows."""
    accent = "#ef4444" if is_active else "#22c55e"

    if is_active:
        caption = "Chance alert ENDS within"
        items = [
            ("15 min", probabilities.get("prob_ends_within_15m", 0.0)),
            ("30 min", probabilities.get("prob_ends_within_30m", 0.0)),
            ("60 min", probabilities.get("prob_ends_within_60m", 0.0)),
        ]
    else:
        caption = "Chance next alert STARTS within"
        items = [
            ("1 hour", probabilities.get("prob_starts_within_1h", 0.0)),
            ("3 hours", probabilities.get("prob_starts_within_3h", 0.0)),
            ("6 hours", probabilities.get("prob_starts_within_6h", 0.0)),
        ]

    st.markdown(
        f"<div style='color:#94a3b8; font-size:12px; text-transform:uppercase; "
        f"letter-spacing:0.5px; margin:4px 0 8px;'>{caption}</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="small")
    for col, (label, prob) in zip(cols, items):
        pct = max(0, min(100, round(prob * 100)))
        with col:
            st.markdown(f"""
            <div style="background-color: #1e242d; border-radius: 8px; padding: 12px 8px; text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #f8fafc;">{pct}%</div>
                <div style="color: #94a3b8; font-size: 12px; margin-top: 2px;">{label}</div>
                <div style="background-color: #334155; border-radius: 4px; height: 5px; margin-top: 8px; overflow: hidden;">
                    <div style="background-color: {accent}; width: {pct}%; height: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def create_remaining_time_curve(
    elapsed_min: float,
    median_remaining: float,
    p10: float,
    p90: float,
    accent: str = "#38bdf8",
    accent_fill: str = "rgba(56, 189, 248, 0.15)",
    unit: str = "min",
) -> go.Figure:
    """Generates an illustrative conditional remaining-time distribution curve.
    `unit='hours'` rescales minute-valued inputs for display (Task B's
    ranges run into the thousands of minutes, which reads better in hours)."""
    scale = 60.0 if unit == "hours" else 1.0
    unit_label = "Hours" if unit == "hours" else "Minutes"
    unit_suffix = "h" if unit == "hours" else "m"
    median_s, p10_s, p90_s = median_remaining / scale, p10 / scale, p90 / scale

    # Approximate smooth lognormal distribution matching P10, P50, P90
    x = np.linspace(0, max(2.0, p90_s * 1.5), 100)
    sigma = (p90_s - p10_s) / 2.56 if (p90_s - p10_s) > 0 else 0.25
    y = np.exp(-0.5 * ((x - median_s) / (sigma + 1e-5)) ** 2)
    y = y / y.max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color=accent, width=3),
        fill="tozeroy",
        fillcolor=accent_fill,
        name="Likelihood Density",
    ))

    fig.add_vline(x=median_s, line_dash="solid", line_color="#f59e0b", annotation_text=f"Median ({median_s:.1f}{unit_suffix})", annotation_position="top")
    fig.add_vrect(x0=p10_s, x1=p90_s, fillcolor="rgba(245, 158, 11, 0.1)", line_width=0, annotation_text="P10–P90 Range", annotation_position="top left")

    fig.update_layout(
        paper_bgcolor="#1f242d",
        plot_bgcolor="#1f242d",
        height=200,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis=dict(title=f"Remaining {unit_label}", color="#94a3b8", showgrid=True, gridcolor="#334155"),
        yaxis=dict(showticklabels=False, showgrid=False),
        showlegend=False,
    )
    return fig

