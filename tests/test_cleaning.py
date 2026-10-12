"""Tests for data.cleaning: oblast name mapping, gap-episode construction,
and the chronological train/test split."""
import pandas as pd

from alertpred.data.cleaning import _map_oblast_name, build_gap_episodes, create_time_split


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class TestMapOblastName:
    def test_known_oblast_name_maps_to_its_configured_id_and_english_name(self):
        # Arrange / Act
        region_id, region_en = _map_oblast_name("Харківська область")

        # Assert
        assert region_id == "kharkiv"
        assert region_en == "Kharkiv Oblast"

    def test_unknown_name_falls_back_to_a_slugified_id_and_keeps_original_as_english(self):
        # Arrange
        unknown_name = "Невідома Область"

        # Act
        region_id, region_en = _map_oblast_name(unknown_name)

        # Assert
        assert region_id == "невідома_область"
        assert region_en == unknown_name

    def test_apostrophes_are_stripped_from_the_fallback_slug(self):
        # Arrange: an unmapped name containing an apostrophe
        unknown_name = "П'ятихатська Область"

        # Act
        region_id, _ = _map_oblast_name(unknown_name)

        # Assert
        assert "'" not in region_id


class TestBuildGapEpisodes:
    def test_two_consecutive_alerts_produce_one_gap_with_correct_duration(self):
        # Arrange: alert 1 ends at 10:00, alert 2 starts at 10:30 -> 30 min gap
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv", "kyiv"],
            "region_en": ["Kyiv Oblast", "Kyiv Oblast"],
            "started_at": [_ts("2026-01-01 09:00:00"), _ts("2026-01-01 10:30:00")],
            "finished_at": [_ts("2026-01-01 10:00:00"), _ts("2026-01-01 11:00:00")],
            "duration_min": [60.0, 30.0],
        })

        # Act
        df_gaps = build_gap_episodes(df_alerts)

        # Assert
        assert len(df_gaps) == 1
        assert df_gaps.loc[0, "gap_duration_min"] == 30.0
        assert df_gaps.loc[0, "prev_alert_duration_min"] == 60.0

    def test_a_single_alert_in_a_region_produces_no_gap(self):
        # Arrange: only one episode for this region, nothing to measure a gap against
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv"],
            "region_en": ["Kyiv Oblast"],
            "started_at": [_ts("2026-01-01 09:00:00")],
            "finished_at": [_ts("2026-01-01 10:00:00")],
            "duration_min": [60.0],
        })

        # Act
        df_gaps = build_gap_episodes(df_alerts)

        # Assert
        assert df_gaps.empty

    def test_overlapping_alerts_produce_no_negative_gap(self):
        # Arrange: the second alert starts before the first one ends
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv", "kyiv"],
            "region_en": ["Kyiv Oblast", "Kyiv Oblast"],
            "started_at": [_ts("2026-01-01 09:00:00"), _ts("2026-01-01 09:30:00")],
            "finished_at": [_ts("2026-01-01 10:00:00"), _ts("2026-01-01 10:30:00")],
            "duration_min": [60.0, 60.0],
        })

        # Act
        df_gaps = build_gap_episodes(df_alerts)

        # Assert: a negative gap_duration_min must be discarded, not kept
        assert df_gaps.empty


class TestCreateTimeSplit:
    def test_default_quantile_cutoff_respects_the_requested_train_ratio(self):
        # Arrange: 10 evenly-spaced rows, 80% train ratio
        df = pd.DataFrame({
            "started_at": [_ts("2026-01-01") + pd.Timedelta(days=i) for i in range(10)],
        })

        # Act
        train_df, test_df = create_time_split(df, timestamp_col="started_at", train_ratio=0.8)

        # Assert: roughly 8 rows in train, the remainder in test
        assert len(train_df) >= 7
        assert len(train_df) + len(test_df) == 10
