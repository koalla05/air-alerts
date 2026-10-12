"""Explicit data-leakage tests (PLAN.md Section 4.3: "Build features with an
explicit as_of_ts and test this with unit tests (e.g. shift check)").

The core technique throughout is a *shift check*: build a feature at a fixed
as_of_ts, then take the exact same input and shift one event from the past
to the future (or vice versa) relative to that as_of_ts. A leakage-safe
feature must change only when the shifted event crosses into the past; it
must be completely blind to anything still in the future.
"""
import pandas as pd
import pytest

from alertpred.data.cleaning import create_time_split
from alertpred.features.builder import (
    compute_neighbor_active_features,
    extract_point_in_time_features,
    sample_as_of_alert_moments,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class TestNeighborFeatureShiftCheck:
    """compute_neighbor_active_features must only see a neighbor's episode
    if that episode had already started by as_of_ts."""

    def test_future_neighbor_episode_is_not_counted_as_active(self):
        # Arrange: a sample for "kyiv" at a fixed as_of_ts, and a neighbor
        # ("kyiv_oblast") whose only episode starts *after* that as_of_ts.
        as_of = _ts("2026-01-01 12:00:00")
        samples = pd.DataFrame({"region_id": ["kyiv"], "as_of_ts": [as_of]})
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv_oblast"],
            "started_at": [as_of + pd.Timedelta(hours=1)],   # future relative to as_of
            "finished_at": [as_of + pd.Timedelta(hours=2)],
        })
        neighbors_map = {"kyiv": ["kyiv_oblast"]}

        # Act
        result = compute_neighbor_active_features(samples, df_alerts, neighbors_map)

        # Assert: a neighbor event that hasn't happened yet must not leak in
        assert result.loc[0, "neighbors_active_count"] == 0
        assert result.loc[0, "neighbors_active_frac"] == 0.0
        assert result.loc[0, "national_active_count"] == 0

    def test_shifting_the_same_episode_into_the_past_flips_the_feature(self):
        # Arrange: identical setup, but the neighbor's episode now spans
        # the as_of_ts (started in the past, still ongoing at as_of).
        as_of = _ts("2026-01-01 12:00:00")
        samples = pd.DataFrame({"region_id": ["kyiv"], "as_of_ts": [as_of]})
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv_oblast"],
            "started_at": [as_of - pd.Timedelta(hours=1)],   # now in the past
            "finished_at": [as_of + pd.Timedelta(hours=1)],  # still active at as_of
        })
        neighbors_map = {"kyiv": ["kyiv_oblast"]}

        # Act
        result = compute_neighbor_active_features(samples, df_alerts, neighbors_map)

        # Assert: once the event starts in the past, it's correctly visible
        assert result.loc[0, "neighbors_active_count"] == 1
        assert result.loc[0, "neighbors_active_frac"] == 1.0

    def test_neighbor_episode_that_already_ended_is_not_active(self):
        # Arrange: neighbor's episode is entirely in the past and already over.
        as_of = _ts("2026-01-01 12:00:00")
        samples = pd.DataFrame({"region_id": ["kyiv"], "as_of_ts": [as_of]})
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv_oblast"],
            "started_at": [as_of - pd.Timedelta(hours=3)],
            "finished_at": [as_of - pd.Timedelta(hours=1)],  # ended before as_of
        })
        neighbors_map = {"kyiv": ["kyiv_oblast"]}

        # Act
        result = compute_neighbor_active_features(samples, df_alerts, neighbors_map)

        # Assert
        assert result.loc[0, "neighbors_active_count"] == 0

    def test_region_with_no_neighbors_gets_zero_fraction_not_nan(self):
        # Arrange: a region absent from neighbors_map (e.g. an island case)
        as_of = _ts("2026-01-01 12:00:00")
        samples = pd.DataFrame({"region_id": ["sevastopol"], "as_of_ts": [as_of]})
        df_alerts = pd.DataFrame({"region_id": [], "started_at": [], "finished_at": []})

        # Act
        result = compute_neighbor_active_features(samples, df_alerts, {"sevastopol": []})

        # Assert: 0/0 must resolve to 0.0, not NaN, so downstream models don't choke
        assert result.loc[0, "neighbors_active_frac"] == 0.0


class TestPointInTimeRollingCountShiftCheck:
    """extract_point_in_time_features' alerts_last_6h/24h must only count
    history strictly before as_of_ts."""

    def test_alert_starting_after_as_of_is_excluded_from_the_count(self):
        # Arrange: a sample at as_of_ts, with the only historical row for the
        # same region starting *after* as_of_ts (i.e. still in the future).
        as_of = _ts("2026-03-01 10:00:00")
        samples = pd.DataFrame({"region_id": ["kharkiv"], "as_of_ts": [as_of]})
        history = pd.DataFrame({
            "region_id": ["kharkiv"],
            "started_at": [as_of + pd.Timedelta(hours=2)],
        })

        # Act
        result = extract_point_in_time_features(samples, history)

        # Assert
        assert result.loc[0, "alerts_last_6h"] == 0
        assert result.loc[0, "alerts_last_24h"] == 0

    def test_shifting_the_same_alert_into_the_past_makes_it_count(self):
        # Arrange: identical as_of_ts, but the historical alert now starts
        # 2 hours *before* as_of_ts instead of after it.
        as_of = _ts("2026-03-01 10:00:00")
        samples = pd.DataFrame({"region_id": ["kharkiv"], "as_of_ts": [as_of]})
        history = pd.DataFrame({
            "region_id": ["kharkiv"],
            "started_at": [as_of - pd.Timedelta(hours=2)],
        })

        # Act
        result = extract_point_in_time_features(samples, history)

        # Assert: now within the 6h and 24h lookback windows
        assert result.loc[0, "alerts_last_6h"] == 1
        assert result.loc[0, "alerts_last_24h"] == 1

    def test_alert_in_a_different_region_never_counts(self):
        # Arrange: a historical alert well within the lookback window, but
        # for a different region than the sample's own.
        as_of = _ts("2026-03-01 10:00:00")
        samples = pd.DataFrame({"region_id": ["kharkiv"], "as_of_ts": [as_of]})
        history = pd.DataFrame({
            "region_id": ["odesa"],
            "started_at": [as_of - pd.Timedelta(minutes=30)],
        })

        # Act
        result = extract_point_in_time_features(samples, history)

        # Assert
        assert result.loc[0, "alerts_last_6h"] == 0


class TestAsOfSamplingNeverExceedsKnownDuration:
    """sample_as_of_alert_moments must never manufacture a snapshot moment
    beyond the episode's actual (historically known) end."""

    def test_every_sampled_elapsed_time_is_strictly_within_the_true_duration(self):
        # Arrange: one 47-minute episode
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv"],
            "region_en": ["Kyiv Oblast"],
            "started_at": [_ts("2026-01-01 00:00:00")],
            "duration_min": [47.0],
        })

        # Act
        samples = sample_as_of_alert_moments(df_alerts)

        # Assert: no snapshot's elapsed time reaches or exceeds the true duration
        assert (samples["elapsed_min"] < samples["total_duration_min"]).all()
        # and the remaining-time target is always strictly positive
        assert (samples["target_remaining_min"] > 0).all()

    def test_as_of_timestamp_never_lands_after_the_episode_actually_ended(self):
        # Arrange
        start = _ts("2026-01-01 00:00:00")
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv"],
            "region_en": ["Kyiv Oblast"],
            "started_at": [start],
            "duration_min": [47.0],
        })
        true_end = start + pd.Timedelta(minutes=47)

        # Act
        samples = sample_as_of_alert_moments(df_alerts)

        # Assert
        assert (samples["as_of_ts"] < true_end).all()


class TestChronologicalSplitHasNoLeakage:
    """create_time_split must give train only the past and test only the
    future, with no row duplicated or dropped across the boundary."""

    def test_train_rows_are_strictly_before_cutoff_and_test_rows_on_or_after(self):
        # Arrange: 10 episodes spread one day apart, explicit cutoff midway
        df = pd.DataFrame({
            "started_at": [_ts("2026-01-01") + pd.Timedelta(days=i) for i in range(10)],
            "value": range(10),
        })
        cutoff = _ts("2026-01-06")

        # Act
        train_df, test_df = create_time_split(df, timestamp_col="started_at", split_date=cutoff)

        # Assert
        assert (train_df["started_at"] < cutoff).all()
        assert (test_df["started_at"] >= cutoff).all()

    def test_split_partitions_every_row_exactly_once(self):
        # Arrange
        df = pd.DataFrame({
            "started_at": [_ts("2026-01-01") + pd.Timedelta(days=i) for i in range(10)],
            "value": range(10),
        })

        # Act
        train_df, test_df = create_time_split(df, timestamp_col="started_at", train_ratio=0.7)

        # Assert: union reconstructs the original row count, no overlap
        assert len(train_df) + len(test_df) == len(df)
        assert set(train_df["value"]).isdisjoint(set(test_df["value"]))
