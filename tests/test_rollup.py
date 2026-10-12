"""Tests for data.rollup.merge_intervals_to_episodes - the interval-merge
logic that rolls raion/hromada alerts up to oblast-level episodes."""
import pandas as pd

from alertpred.data.rollup import merge_intervals_to_episodes


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _rec(start: str, end: str, is_native_oblast: bool = True) -> dict:
    return {"started_at": _ts(start), "finished_at": _ts(end), "is_native_oblast": is_native_oblast}


class TestBasicMerging:
    def test_empty_input_returns_empty_output(self):
        # Arrange / Act
        result = merge_intervals_to_episodes([])

        # Assert
        assert result == []

    def test_touching_intervals_are_merged_into_one_episode(self):
        # Arrange: second interval starts exactly when the first ends
        records = [
            _rec("2026-01-01 10:00:00", "2026-01-01 11:00:00"),
            _rec("2026-01-01 11:00:00", "2026-01-01 12:00:00"),
        ]

        # Act
        episodes = merge_intervals_to_episodes(records)

        # Assert
        assert len(episodes) == 1
        assert episodes[0]["started_at"] == _ts("2026-01-01 10:00:00")
        assert episodes[0]["finished_at"] == _ts("2026-01-01 12:00:00")
        assert episodes[0]["duration_min"] == 120.0

    def test_overlapping_intervals_are_merged_to_the_latest_end(self):
        # Arrange: second interval starts before the first one ends
        records = [
            _rec("2026-01-01 10:00:00", "2026-01-01 11:00:00"),
            _rec("2026-01-01 10:30:00", "2026-01-01 11:30:00"),
        ]

        # Act
        episodes = merge_intervals_to_episodes(records)

        # Assert
        assert len(episodes) == 1
        assert episodes[0]["finished_at"] == _ts("2026-01-01 11:30:00")

    def test_gapped_intervals_stay_as_separate_episodes(self):
        # Arrange: a real quiet gap between the two intervals
        records = [
            _rec("2026-01-01 10:00:00", "2026-01-01 11:00:00"),
            _rec("2026-01-01 14:00:00", "2026-01-01 15:00:00"),
        ]

        # Act
        episodes = merge_intervals_to_episodes(records)

        # Assert
        assert len(episodes) == 2

    def test_out_of_order_input_is_still_merged_correctly(self):
        # Arrange: records passed in reverse chronological order
        records = [
            _rec("2026-01-01 11:00:00", "2026-01-01 12:00:00"),
            _rec("2026-01-01 10:00:00", "2026-01-01 11:00:00"),
        ]

        # Act
        episodes = merge_intervals_to_episodes(records)

        # Assert: still merged into one, sorted result
        assert len(episodes) == 1
        assert episodes[0]["started_at"] == _ts("2026-01-01 10:00:00")


class TestIsRollupFlag:
    def test_merge_of_only_native_oblast_records_is_not_flagged_as_rollup(self):
        # Arrange
        records = [_rec("2026-01-01 10:00:00", "2026-01-01 11:00:00", is_native_oblast=True)]

        # Act
        episodes = merge_intervals_to_episodes(records)

        # Assert
        assert episodes[0]["is_rollup"] is False

    def test_merge_including_any_sub_oblast_record_is_flagged_as_rollup(self):
        # Arrange: one native oblast record merged with one raion-derived record
        records = [
            _rec("2026-01-01 10:00:00", "2026-01-01 11:00:00", is_native_oblast=True),
            _rec("2026-01-01 11:00:00", "2026-01-01 12:00:00", is_native_oblast=False),
        ]

        # Act
        episodes = merge_intervals_to_episodes(records)

        # Assert
        assert len(episodes) == 1
        assert episodes[0]["is_rollup"] is True


class TestCorruptRecordRegression:
    """Regression test for a real bug: a single never-closed source row
    (observed in practice: a hromada-level record spanning 604 days) used to
    chain-merge every real episode between its start and end into one giant
    episode, which then got dropped entirely by the downstream max-duration
    filter - silently destroying up to years of valid data. The fix filters
    individual records longer than `max_single_record_hours` before merging."""

    def test_one_corrupt_long_record_does_not_swallow_surrounding_real_episodes(self):
        # Arrange: a real episode that touches the start of a corrupt,
        # never-closed record (so they'd chain under the old behavior), and
        # a second real episode that falls entirely inside the corrupt
        # record's bogus multi-week span (so it would have been silently
        # absorbed into the chain and lost, never appearing as its own
        # episode) - this is the exact shape of the real Kharkiv bug.
        touches_corrupt_start = _rec("2026-01-01 08:00:00", "2026-01-02 00:00:00")
        corrupt_record = _rec("2026-01-02 00:00:00", "2026-03-01 00:00:00")  # ~58 days, never closed
        swallowed_without_the_fix = _rec("2026-01-20 08:00:00", "2026-01-20 09:00:00")
        records = [touches_corrupt_start, corrupt_record, swallowed_without_the_fix]

        # Sanity-check the premise: without the guard, these three records
        # really do chain-merge into one ~58-day blob (reproducing the bug)
        unguarded = merge_intervals_to_episodes(records, max_single_record_hours=1e9)
        assert len(unguarded) == 1
        assert unguarded[0]["duration_min"] > 24 * 60

        # Act
        episodes = merge_intervals_to_episodes(records, max_single_record_hours=24.0)

        # Assert: the corrupt record is dropped before merging, so the two
        # real episodes survive independently rather than being chained
        # into one multi-month blob (and then discarded downstream)
        assert len(episodes) == 2
        starts = {ep["started_at"] for ep in episodes}
        assert _ts("2026-01-01 08:00:00") in starts
        assert _ts("2026-01-20 08:00:00") in starts
        # none of the surviving episodes should span anywhere near the corrupt window
        assert all(ep["duration_min"] < 24 * 60 for ep in episodes)

    def test_corrupt_record_alone_yields_no_episodes_rather_than_crashing(self):
        # Arrange: only the corrupt record, nothing else
        records = [_rec("2026-01-02 00:00:00", "2026-03-01 00:00:00")]

        # Act
        episodes = merge_intervals_to_episodes(records, max_single_record_hours=24.0)

        # Assert
        assert episodes == []

    def test_a_genuinely_long_but_plausible_record_is_kept(self):
        # Arrange: a single 20-hour record, under the 24h single-record cap
        records = [_rec("2026-01-01 00:00:00", "2026-01-01 20:00:00")]

        # Act
        episodes = merge_intervals_to_episodes(records, max_single_record_hours=24.0)

        # Assert
        assert len(episodes) == 1
        assert episodes[0]["duration_min"] == 20 * 60
