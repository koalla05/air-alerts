from typing import Dict, Any, List
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import datetime


def render_probability_bars(probabilities: Dict[str, float], is_active: bool):
    """Renders calibrated probability progress indicators."""
    st.markdown("#### Forecast Probabilities")
    
    if is_active:
        items = [
            ("Ends within 15 minutes", probabilities.get("prob_ends_within_15m", 0.0)),
            ("Ends within 30 minutes", probabilities.get("prob_ends_within_30m", 0.0)),
            ("Ends within 60 minutes", probabilities.get("prob_ends_within_60m", 0.0)),
        ]
    else:
        items = [
            ("Alert starts within 1 hour", probabilities.get("prob_starts_within_1h", 0.0)),
            ("Alert starts within 3 hours", probabilities.get("prob_starts_within_3h", 0.0)),
            ("Alert starts within 6 hours", probabilities.get("prob_starts_within_6h", 0.0)),
        ]

    for label, prob in items:
        pct = int(prob * 100)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{label}**")
            st.progress(prob)
        with col2:
            st.markdown(f"<h3 style='margin:0; text-align:right;'>{pct}%</h3>", unsafe_allow_html=True)


def create_remaining_time_curve(elapsed_min: float, median_remaining: float, p10: float, p90: float) -> go.Figure:
    """Generates an illustrative conditional remaining-time distribution curve."""
    # Approximate smooth lognormal distribution matching P10, P50, P90
    x = np.linspace(0, max(120, p90 * 1.5), 100)
    # Using simple smoothed curve peaked at median
    sigma = (p90 - p10) / 2.56 if (p90 - p10) > 0 else 15.0
    y = np.exp(-0.5 * ((x - median_remaining) / (sigma + 1e-5)) ** 2)
    y = y / y.max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color="#38bdf8", width=3),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.15)",
        name="Likelihood Density",
    ))

    # Add vertical lines for P10, P50, P90
    fig.add_vline(x=median_remaining, line_dash="solid", line_color="#f59e0b", annotation_text=f"Median ({int(median_remaining)}m)", annotation_position="top")
    fig.add_vrect(x0=p10, x1=p90, fillcolor="rgba(245, 158, 11, 0.1)", line_width=0, annotation_text="P10–P90 Range", annotation_position="top left")

    fig.update_layout(
        paper_bgcolor="#1f242d",
        plot_bgcolor="#1f242d",
        height=180,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis=dict(title="Remaining Minutes", color="#94a3b8", showgrid=True, gridcolor="#334155"),
        yaxis=dict(showticklabels=False, showgrid=False),
        showlegend=False,
    )
    return fig

