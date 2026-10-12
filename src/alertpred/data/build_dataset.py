"""Convenience wrapper around the data pipeline walked through step-by-step
in notebook 01 (load raw CSV -> rollup to oblast level -> backfill via
alerts.in.ua -> merge -> build gap episodes), for consumers (like notebook
02) that just want the resulting table without repeating that walkthrough.

Not wired into `train.py` / the Streamlit app yet - those still use the
narrower `clean_alerts` pipeline until the regime-handling comparison in
notebook 02 settles which approach to productionize.
"""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..config import RAW_DATA_PATH
from ..ingestion.alerts_client import get_api_token
from ..ingestion.backfill import fetch_all_region_histories, merge_with_existing
from .cleaning import (
    build_gap_episodes,
    build_oblast_rollup_episodes,
    load_and_clean_volunteer_data,
    load_raw_data,
)


def build_master_alerts_dataset(
    raw_data_path: Optional[Path] = None,
    use_api_backfill: bool = True,
    api_period: str = "month_ago",
    use_volunteer_gapfill: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_alerts, df_gaps) - the bias-corrected, oblast-level,
    rollup-aware episode table (see data.rollup) and its derived quiet-gap
    table, backfilled with recent alerts.in.ua history.

    `use_volunteer_gapfill`: the volunteer dataset (see
    `cleaning.load_and_clean_volunteer_data`) measures alerts differently
    enough from the official data (many more, much shorter episodes that
    don't even align in time - see that function's docstring) that it's
    used *only* to fill the specific date gap left between the GitHub CSV's
    last record and the alerts.in.ua backfill's earliest one, not blended
    across the whole timeline where it would be mixing two different
    measurement methodologies into one distribution.
    """
    df_raw = load_raw_data(raw_data_path or RAW_DATA_PATH)
    df_github = build_oblast_rollup_episodes(df_raw)

    df_api = pd.DataFrame()
    if use_api_backfill and get_api_token():
        df_api = fetch_all_region_histories(period=api_period, verbose=verbose)

    df_alerts = df_github
    if use_volunteer_gapfill and not df_api.empty:
        gap_start, gap_end = df_github["started_at"].max(), df_api["started_at"].min()
        if gap_end > gap_start:
            df_volunteer = load_and_clean_volunteer_data()
            df_gapfill = df_volunteer[
                (df_volunteer["started_at"] >= gap_start) & (df_volunteer["started_at"] < gap_end)
            ]
            if verbose:
                print(f"Filling {gap_end - gap_start} gap ({gap_start.date()} to {gap_end.date()}) "
                      f"with {len(df_gapfill):,} volunteer-sourced episodes")
            df_alerts = merge_with_existing(df_alerts, df_gapfill)

    df_alerts = merge_with_existing(df_alerts, df_api)
    df_gaps = build_gap_episodes(df_alerts)
    return df_alerts, df_gaps
