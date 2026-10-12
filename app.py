import sys
from pathlib import Path
import datetime
import streamlit as st
import pandas as pd

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alertpred.config import (
    REGION_MAPPING,
    MODELS_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
)
from alertpred.models.quantile_lgbm import AlertPredictionPipeline
from alertpred.ingestion.live import get_live_alerts_status, ISO_MAPPING
from alertpred.ingestion.alerts_regions import OBLAST_UID
from alertpred.ingestion.alerts_client import fetch_region_history, get_api_token
from alertpred.ingestion.backfill import parse_oblast_episodes_from_payload
from alertpred.app.map import create_raion_map
from alertpred.app.components import render_probability_bars, create_remaining_time_curve
from alertpred.app.stats import render_stats_tab
from alertpred.geo.raions import RAIONS_BY_OBLAST, RAION_BY_ID, API_NAME_TO_RAION_ID


# Set Streamlit Page Config
st.set_page_config(
    page_title="Ukraine Air Raid Predictor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_trained_pipeline():
    """Load serialized LightGBM & Baseline pipeline."""
    model_path = MODELS_DIR / "alertpred_pipeline.joblib"
    if not model_path.exists():
        st.error(f"Trained model not found at {model_path}. Please run `python -m alertpred.train` first.")
        st.stop()
    return AlertPredictionPipeline.load(model_path)


pipeline = load_trained_pipeline()


@st.cache_data(ttl=600)
def load_last_alert_end_by_region():
    """Last known alert end per region, from the processed episode table
    (written by `train.py`), used to compute a real elapsed-quiet time
    instead of the rough placeholder `get_live_alerts_status` otherwise
    falls back to (flat 4h for Kyiv, 8h for everyone else - not real data)."""
    path = PROCESSED_DATA_DIR / "cleaned_alerts.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["region_id", "finished_at"])
    return df.groupby("region_id")["finished_at"].max().to_dict()


last_alert_end_by_region = load_last_alert_end_by_region()


@st.cache_data(ttl=900)
def fetch_fresh_quiet_status(region_id: str, raion_id: "str | None"):
    """
    Live "last known alert end" for the selected oblast and, if one is
    selected, a specific raion within it - replacing the training-snapshot
    fallback entirely rather than patching over it, and giving raions real
    tracking for the first time (they previously always showed a flat 8h
    placeholder, never real data).

    `last_alert_end_by_region` (loaded from `data/processed/cleaned_alerts.parquet`)
    is a training-time snapshot, only as fresh as the last `train.py` run,
    and we've already hit real bugs that made it wrong - presenting a
    specific-looking "quiet for X hours" sourced from it is false precision
    dressed up as fact. alerts.in.ua's history endpoint only looks back
    ~30 days, so it can't replace the snapshot's *depth*, but within that
    window it's live and verified, so it's preferred whenever available.
    When the live window comes back with nothing, that's still real
    information - "no alert in at least 30 days" - just reported honestly
    as a lower bound instead of silently substituting a snapshot value of
    unknown age.

    Returns (oblast_last_end, oblast_is_lower_bound, raion_last_end, raion_is_lower_bound).
    A `None` end (only possible when there's no token / the call fails)
    means "no live data at all" - the caller falls back to the training
    snapshot in that case, same as before.
    """
    uid = OBLAST_UID.get(region_id)
    if uid is None or not get_api_token():
        return None, False, None, False
    try:
        payload = fetch_region_history(uid, cache_ttl_sec=900)
    except Exception:
        return None, False, None, False

    alert_objs = payload.get("alerts", [])
    thirty_days_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)

    oblast_episodes = parse_oblast_episodes_from_payload(alert_objs, region_id)
    oblast_ends = [ep["finished_at"] for ep in oblast_episodes]
    oblast_end = max(oblast_ends) if oblast_ends else thirty_days_ago
    oblast_is_lower_bound = not oblast_ends

    raion_end, raion_is_lower_bound = None, False
    if raion_id is not None:
        raion_ends = [
            pd.to_datetime(a["finished_at"], utc=True)
            for a in alert_objs
            if a.get("location_type") == "raion" and a.get("finished_at")
            and API_NAME_TO_RAION_ID.get(a.get("location_title")) == raion_id
        ]
        raion_end = max(raion_ends) if raion_ends else thirty_days_ago
        raion_is_lower_bound = not raion_ends

    return oblast_end, oblast_is_lower_bound, raion_end, raion_is_lower_bound


@st.cache_data(ttl=600)
def load_alerts_df():
    """Full processed episode table, for the Statistics tab."""
    path = PROCESSED_DATA_DIR / "cleaned_alerts.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# Sidebar: Region Selection & Simulation Controls
st.sidebar.title("Region Settings")

# Build display options for regions
region_options = {}
for ukr_name, meta in REGION_MAPPING.items():
    if meta["id"] in ISO_MAPPING:
        label = f"{meta['en']} ({ukr_name})"
        region_options[label] = meta["id"]

selected_label = st.sidebar.selectbox(
    "Select Oblast to Inspect:",
    options=list(region_options.keys()),
    index=0,
)
selected_region_id = region_options[selected_label]
selected_meta = next(v for v in REGION_MAPPING.values() if v["id"] == selected_region_id)

# Raion-level selection, scoped to the chosen oblast. The model is still
# trained on oblast-level history only (raion-level alerts only became the
# dominant reporting granularity recently - see the EDA notebook), so a
# raion selection reuses its parent oblast's trained model, anchored to the
# raion's own live elapsed time. Proper raion-level training is future work.
raion_ids_in_oblast = RAIONS_BY_OBLAST.get(selected_region_id, [])
raion_options = {"Whole Oblast": None}
for rid in raion_ids_in_oblast:
    meta = RAION_BY_ID[rid]
    if not meta["is_pseudo_oblast"]:
        raion_options[f"{meta['name_en']} ({meta['name_uk']})"] = rid

selected_raion_label = st.sidebar.selectbox(
    "Select Raion (optional, true alert granularity):",
    options=list(raion_options.keys()),
    index=0,
)
selected_raion_id = raion_options[selected_raion_label]
if selected_raion_id is not None:
    st.sidebar.caption("Prediction uses the parent oblast's model, anchored to this raion's live elapsed time.")

st.sidebar.markdown("---")
st.sidebar.subheader("Live & Simulation Mode")

# Simulation override toggle
enable_sim_override = st.sidebar.checkbox("Simulate Custom State", value=False)
if enable_sim_override:
    sim_is_alert = st.sidebar.toggle("Active Alert in Selected Oblast", value=True)
    if sim_is_alert:
        sim_elapsed = st.sidebar.slider("Minutes Elapsed in Alert:", min_value=1, max_value=180, value=30)
        sim_quiet = 0.0
    else:
        sim_elapsed = 0.0
        sim_quiet = st.sidebar.slider("Hours Elapsed Since Last Alert:", min_value=1, max_value=72, value=4) * 60.0
else:
    sim_is_alert = None
    sim_elapsed = 0.0
    sim_quiet = 0.0

# Prefer live data over the training-time snapshot for the selected
# oblast (and raion, if one is selected) - see fetch_fresh_quiet_status's
# docstring for why a stale fallback isn't good enough on its own.
oblast_end, oblast_is_lower_bound, raion_end, raion_is_lower_bound = fetch_fresh_quiet_status(
    selected_region_id, selected_raion_id
)
if oblast_end is not None:
    last_alert_end_by_region = {**last_alert_end_by_region, selected_region_id: oblast_end}

# Fetch Live Status
live_status = get_live_alerts_status(last_alert_end_by_region=last_alert_end_by_region)

# Raion quiet-time has no equivalent input into get_live_alerts_status
# (it always used a flat 480min placeholder for raions - see the chat
# discussion), so it's set directly here from the live fetch above.
if (
    selected_raion_id is not None
    and raion_end is not None
    and selected_raion_id in live_status["raions"]
    and not live_status["raions"][selected_raion_id]["is_active"]
):
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    live_status["raions"][selected_raion_id]["elapsed_quiet_min"] = max(
        0.0, (now_utc - raion_end).total_seconds() / 60.0
    )

# Apply simulation override if active (to the raion if one is selected, else the whole oblast)
if enable_sim_override:
    sim_state = {
        "is_active": sim_is_alert,
        "elapsed_min": float(sim_elapsed),
        "elapsed_quiet_min": float(sim_quiet),
    }
    if selected_raion_id is not None and selected_raion_id in live_status["raions"]:
        live_status["raions"][selected_raion_id] = sim_state
    elif selected_region_id in live_status["regions"]:
        live_status["regions"][selected_region_id] = sim_state
    # Recompute active count
    live_status["total_active_regions"] = sum(1 for r in live_status["regions"].values() if r["is_active"])

# Header National Overview
st.title("Ukraine Air Raid Alerts: Duration & Expiration Predictor")

tab_live, tab_stats = st.tabs(["🚨 Live Dashboard", "📊 Statistics"])

with tab_live:
    now_kyiv = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=3)))
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.metric(
            label="Active Alerts",
            value=f"{live_status['total_active_regions']} / {len(live_status['regions'])} Oblasts",
            delta="Elevated Activity" if live_status["total_active_regions"] > 4 else "Calm",
            delta_color="inverse",
        )
    with col_m2:
        st.metric(
            label="Status Mode",
            value="Live API" if live_status["is_live_api"] else "Demo Simulation",
        )
    with col_m3:
        st.metric(
            label="Last Updated",
            value=now_kyiv.strftime("%H:%M:%S (Kyiv)"),
        )

    if not live_status["is_live_api"]:
        st.info("ℹ️ Running in **Demo Simulation Mode** (realistic snapshot). To connect to live production sirens, set `ALERTS_IN_UA_TOKEN`.", icon="📡")

    # Main Content Layout: Map on the Left (60%), Forecast Panel on the Right (40%)
    col_map, col_forecast = st.columns([1.3, 1.0], gap="medium")

    with col_map:
        st.subheader("National Siren Map (by Raion)")
        map_fig = create_raion_map(live_status, selected_raion_id)
        st.plotly_chart(map_fig, width='stretch', config={"displayModeBar": False})

    with col_forecast:
        # Raion-level live status is the true, granular signal; predictions
        # still come from the parent oblast's trained model (see sidebar
        # caption) until raion-level training is built (tracked as future work).
        if selected_raion_id is not None:
            region_state = live_status["raions"].get(selected_raion_id, {"is_active": False, "elapsed_min": 0, "elapsed_quiet_min": 480})
            display_name = f"{selected_raion_label.split(' (')[0]} ({selected_meta['en']})"
            quiet_is_lower_bound = raion_is_lower_bound and raion_end is not None
        else:
            region_state = live_status["regions"].get(selected_region_id, {"is_active": False, "elapsed_min": 0, "elapsed_quiet_min": 240})
            display_name = selected_meta["en"]
            quiet_is_lower_bound = oblast_is_lower_bound and oblast_end is not None

        is_active = region_state["is_active"]
        elapsed_min = region_state["elapsed_min"]
        elapsed_quiet_min = region_state["elapsed_quiet_min"]
        if enable_sim_override:
            quiet_is_lower_bound = False  # an explicit simulated value is exact by definition

        st.subheader(display_name)
        if selected_raion_id is not None:
            st.caption(f"Prediction model: {selected_meta['en']} (oblast-level - no raion-level model yet)")

        if is_active:
            st.error(f"🚨 **ACTIVE AIR RAID ALERT** · Running for **{int(elapsed_min)} minutes**")

            # Inference for Active Alert
            pred = pipeline.predict_active_alert(selected_region_id, elapsed_min=elapsed_min, live_status=live_status)
            median_rem = pred["predicted_remaining_median_min"]
            p10, p90 = pred["predicted_range_p10_p90_min"]

            # Estimated clock time of alert end in Kyiv time
            est_end_dt = now_kyiv + datetime.timedelta(minutes=median_rem)

            # Primary Metrics Card
            st.markdown(f"""
            <div style="background-color: #1e242d; padding: 18px; border-radius: 8px; border-left: 5px solid #ef4444; margin-bottom: 15px;">
                <div style="color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Remaining Duration</div>
                <div style="font-size: 28px; font-weight: 700; color: #f8fafc; margin-top: 4px;">
                    ~{median_rem:.0f} min <span style="font-size: 16px; font-weight: 400; color: #94a3b8;">(est. end at {est_end_dt.strftime('%H:%M')})</span>
                </div>
                <div style="color: #cbd5e1; font-size: 13px; margin-top: 6px;">
                    Likely interval (P10–P90): <strong>{p10:.0f} – {p90:.0f} minutes</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability Forecast Bars
            render_probability_bars(pred["probabilities"], is_active=True)

            # Likelihood distribution curve
            st.markdown("#### Remaining Time Distribution")
            chart_fig = create_remaining_time_curve(
                elapsed_min, median_rem, p10, p90,
                accent="#ef4444", accent_fill="rgba(239, 68, 68, 0.15)",
            )
            st.plotly_chart(chart_fig, width='stretch')

        else:
            quiet_hours = elapsed_quiet_min / 60.0
            bound_suffix = "+" if quiet_is_lower_bound else ""
            st.success(f"✅ **NO ACTIVE ALERT** · Quiet for **{quiet_hours:.1f}{bound_suffix} hours**")
            if quiet_is_lower_bound:
                st.caption("No alert found in alerts.in.ua's ~30-day history window - this is a lower bound, not an exact time.")

            # Inference for Quiet Period
            pred = pipeline.predict_quiet_region(selected_region_id, elapsed_quiet_min=elapsed_quiet_min, live_status=live_status)
            median_gap = pred["predicted_next_alert_gap_median_min"]
            p10, p90 = pred["predicted_range_p10_p90_min"]

            est_next_dt = now_kyiv + datetime.timedelta(minutes=median_gap)

            st.markdown(f"""
            <div style="background-color: #1e242d; padding: 18px; border-radius: 8px; border-left: 5px solid #22c55e; margin-bottom: 15px;">
                <div style="color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Time to Next Siren</div>
                <div style="font-size: 28px; font-weight: 700; color: #f8fafc; margin-top: 4px;">
                    ~{median_gap/60.0:.1f} hours <span style="font-size: 16px; font-weight: 400; color: #94a3b8;">({median_gap:.0f} min)</span>
                </div>
                <div style="color: #cbd5e1; font-size: 13px; margin-top: 6px;">
                    Uncertainty range (P10–P90): <strong>{p10/60.0:.1f} – {p90/60.0:.1f} hours</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability Forecast Bars
            render_probability_bars(pred["probabilities"], is_active=False)

            # Likelihood distribution curve
            st.markdown("#### Time to Next Alert Distribution")
            chart_fig = create_remaining_time_curve(
                elapsed_quiet_min, median_gap, p10, p90,
                accent="#22c55e", accent_fill="rgba(34, 197, 94, 0.15)", unit="hours",
            )
            st.plotly_chart(chart_fig, width='stretch')

with tab_stats:
    df_alerts_full = load_alerts_df()
    render_stats_tab(df_alerts_full, REGION_MAPPING)

# Footer with Safety Disclaimer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 12px; line-height: 1.6;">
    ⚠️ <strong>Important Safety Disclaimer:</strong> This application is an experimental machine learning research project.
    Predictions are purely statistical estimations based on historical siren data (2022–2026) and <strong>must never be used for civil defense, shelter decisions, or personal safety</strong>.
    Always follow official notifications from the State Emergency Service of Ukraine and local civil defense authorities.<br/>
    Prediction engine: LightGBM Quantile Regression &nbsp;·&nbsp;
    Historical Data: <a href="https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset" target="_blank" style="color: #38bdf8;">Ukrainian Air Raid Sirens Dataset (Vadimkin)</a>
</div>
""", unsafe_allow_html=True)

