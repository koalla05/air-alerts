"""Tests for features.builder: recency weighting and as-of sampling targets."""
import pandas as pd
import pytest

from alertpred.features.builder import (
    compute_recency_weight,
    compute_time_bucket_features,
    sample_as_of_alert_moments,
    sample_as_of_gap_moments,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class TestComputeTimeBucketFeatures:
    def test_bucket_is_based_on_kyiv_local_time_not_utc(self):
        # Arrange: 03:00 UTC is 05:00 in Kyiv during winter (UTC+2) - still
        # "night" locally, even though a UTC-only bucket would also say
        # night here (pick a boundary case where UTC and local disagree)
        as_of = pd.Series([_ts("2026-01-01 10:00:00")])  # 12:00 Kyiv local -> afternoon

        # Act
        result = compute_time_bucket_features(as_of)

        # Assert: a naive UTC bucket would call 10:00 UTC "morning"; the
        # correct Kyiv-local bucket is "afternoon"
        assert result.loc[0, "time_of_day"] == "afternoon"

    def test_each_boundary_hour_falls_into_the_expected_bucket(self):
        # Arrange: local midnight, local noon, local 18:00 (winter, UTC+2)
        as_of = pd.Series([
            _ts("2025-12-31 22:00:00"),  # 00:00 Kyiv -> night
            _ts("2026-01-01 10:00:00"),  # 12:00 Kyiv -> afternoon
            _ts("2026-01-01 16:00:00"),  # 18:00 Kyiv -> evening
        ])

        # Act
        result = compute_time_bucket_features(as_of)

        # Assert
        assert list(result["time_of_day"]) == ["night", "afternoon", "evening"]

    def test_day_of_week_matches_the_local_calendar_day(self):
        # Arrange: a known Monday in Kyiv local time
        as_of = pd.Series([_ts("2026-01-05 03:00:00")])

        # Act
        result = compute_time_bucket_features(as_of)

        # Assert
        assert result.loc[0, "day_of_week"] == "Monday"


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
