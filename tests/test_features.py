"""Tests for features.builder: recency weighting and as-of sampling targets."""
import pandas as pd
import pytest

from alertpred.features.builder import (
    compute_recency_weight,
    sample_as_of_alert_moments,
    sample_as_of_gap_moments,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class TestComputeRecencyWeight:
    def test_weight_at_the_reference_timestamp_itself_is_one(self):
        # Arrange
        reference = _ts("2026-01-01")
        timestamps = pd.Series([reference])

        # Act
        weights = compute_recency_weight(timestamps, reference_ts=reference, half_life_days=60.0)

        # Assert
        assert weights.iloc[0] == pytest.approx(1.0)

    def test_weight_exactly_one_half_life_old_is_one_half(self):
        # Arrange
        reference = _ts("2026-03-01")
        half_life = 60.0
        timestamps = pd.Series([reference - pd.Timedelta(days=half_life)])

        # Act
        weights = compute_recency_weight(timestamps, reference_ts=reference, half_life_days=half_life)

        # Assert
        assert weights.iloc[0] == pytest.approx(0.5, rel=1e-6)

    def test_weight_decreases_monotonically_with_age(self):
        # Arrange: three timestamps, increasingly old
        reference = _ts("2026-06-01")
        timestamps = pd.Series([
            reference,
            reference - pd.Timedelta(days=30),
            reference - pd.Timedelta(days=90),
        ])

        # Act
        weights = compute_recency_weight(timestamps, reference_ts=reference, half_life_days=60.0)

        # Assert
        assert weights.iloc[0] > weights.iloc[1] > weights.iloc[2]

    def test_default_reference_is_the_series_own_max_timestamp(self):
        # Arrange: no explicit reference_ts given
        timestamps = pd.Series([_ts("2026-01-01"), _ts("2026-02-01")])

        # Act
        weights = compute_recency_weight(timestamps, half_life_days=60.0)

        # Assert: the latest timestamp in the series gets weight 1.0
        assert weights.iloc[1] == pytest.approx(1.0)
        assert weights.iloc[0] < 1.0


class TestSampleAsOfAlertMoments:
    def test_target_remaining_min_equals_duration_minus_elapsed(self):
        # Arrange: a single 90-minute episode
        df_alerts = pd.DataFrame({
            "region_id": ["odesa"],
            "region_en": ["Odesa Oblast"],
            "started_at": [_ts("2026-01-01 00:00:00")],
            "duration_min": [90.0],
        })

        # Act
        samples = sample_as_of_alert_moments(df_alerts)

        # Assert: for every sampled row, remaining = total - elapsed
        expected = samples["total_duration_min"] - samples["elapsed_min"]
        assert (samples["target_remaining_min"] == expected).all()

    def test_binary_horizon_targets_are_consistent_with_remaining_time(self):
        # Arrange: episode long enough to produce a snapshot with >60min remaining
        df_alerts = pd.DataFrame({
            "region_id": ["odesa"],
            "region_en": ["Odesa Oblast"],
            "started_at": [_ts("2026-01-01 00:00:00")],
            "duration_min": [200.0],
        })

        # Act
        samples = sample_as_of_alert_moments(df_alerts)
        row = samples[samples["elapsed_min"] == 0].iloc[0]

        # Assert: 200 min remaining at t=0, so none of the "ends within" flags should fire
        assert row["target_ends_within_15m"] == 0
        assert row["target_ends_within_30m"] == 0
        assert row["target_ends_within_60m"] == 0

    def test_very_short_episode_yields_only_the_t_equals_zero_snapshot(self):
        # Arrange: a 3-minute episode - only elapsed_steps < 3 qualify, i.e. just t=0
        df_alerts = pd.DataFrame({
            "region_id": ["odesa"],
            "region_en": ["Odesa Oblast"],
            "started_at": [_ts("2026-01-01 00:00:00")],
            "duration_min": [3.0],
        })

        # Act
        samples = sample_as_of_alert_moments(df_alerts)

        # Assert
        assert list(samples["elapsed_min"]) == [0.0]

    def test_empty_input_returns_an_empty_dataframe(self):
        # Arrange
        df_alerts = pd.DataFrame({"region_id": [], "region_en": [], "started_at": [], "duration_min": []})

        # Act
        samples = sample_as_of_alert_moments(df_alerts)

        # Assert
        assert samples.empty


class TestSampleAsOfGapMoments:
    def test_target_remaining_gap_equals_total_minus_elapsed(self):
        # Arrange: one 500-minute quiet gap
        df_gaps = pd.DataFrame({
            "region_id": ["lviv"],
            "region_en": ["Lviv Oblast"],
            "gap_start": [_ts("2026-01-01 00:00:00")],
            "gap_duration_min": [500.0],
            "prev_alert_duration_min": [40.0],
        })

        # Act
        samples = sample_as_of_gap_moments(df_gaps)

        # Assert
        expected = samples["total_gap_min"] - samples["elapsed_quiet_min"]
        assert (samples["target_remaining_gap_min"] == expected).all()
