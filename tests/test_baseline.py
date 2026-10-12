"""Tests for models.baseline: weighted percentile/mean helpers and the
empirical conditional baseline."""
import numpy as np
import pandas as pd
import pytest

from alertpred.models.baseline import (
    EmpiricalConditionalBaseline,
    weighted_mean,
    weighted_percentile,
)


class TestWeightedPercentile:
    def test_uniform_weights_matches_plain_percentile(self):
        # Arrange
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        weights = np.ones_like(values)

        # Act
        result = weighted_percentile(values, weights, 50)

        # Assert: close to the unweighted median (allow for interpolation method differences)
        assert result == pytest.approx(np.percentile(values, 50), abs=5.0)

    def test_heavily_weighting_one_value_pulls_the_median_toward_it(self):
        # Arrange: value 10 carries almost all the weight
        values = np.array([10.0, 20.0, 30.0])
        weights = np.array([100.0, 1.0, 1.0])

        # Act
        result = weighted_percentile(values, weights, 50)

        # Assert
        assert result == pytest.approx(10.0, abs=1.0)

    def test_empty_input_returns_zero_without_crashing(self):
        # Arrange
        values = np.array([])
        weights = np.array([])

        # Act
        result = weighted_percentile(values, weights, 50)

        # Assert
        assert result == 0.0


class TestWeightedMean:
    def test_uniform_weights_matches_plain_mean(self):
        # Arrange
        values = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.ones_like(values)

        # Act
        result = weighted_mean(values, weights)

        # Assert
        assert result == pytest.approx(2.5)

    def test_zero_total_weight_returns_zero_without_dividing_by_zero(self):
        # Arrange
        values = np.array([1.0, 2.0])
        weights = np.array([0.0, 0.0])

        # Act
        result = weighted_mean(values, weights)

        # Assert
        assert result == 0.0


class TestEmpiricalConditionalBaseline:
    def test_predicts_the_conditional_median_remaining_time(self):
        # Arrange: a region where every historical alert lasted exactly 100 min
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv"] * 5,
            "duration_min": [100.0] * 5,
        })
        baseline = EmpiricalConditionalBaseline().fit(df_alerts)

        # Act: query at elapsed=40 -> remaining should be 60 for every surviving episode
        result = baseline.predict_remaining("kyiv", elapsed_min=40.0)

        # Assert
        assert result["p50"] == pytest.approx(60.0)
        assert result["sample_count"] == 5

    def test_unknown_region_falls_back_to_the_global_distribution(self):
        # Arrange
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv"] * 3,
            "duration_min": [50.0, 60.0, 70.0],
        })
        baseline = EmpiricalConditionalBaseline().fit(df_alerts)

        # Act: query a region never seen during fit
        result = baseline.predict_remaining("unseen_region", elapsed_min=0.0)

        # Assert: falls back to the global pool rather than erroring
        assert result["sample_count"] == 3

    def test_recency_weighted_fit_shifts_the_prediction_toward_recent_episodes(self):
        # Arrange: two very different-duration episodes, one far more heavily weighted
        df_alerts = pd.DataFrame({
            "region_id": ["kyiv", "kyiv"],
            "duration_min": [100.0, 1000.0],
            "recency_weight": [0.01, 1.0],
        })
        baseline = EmpiricalConditionalBaseline().fit(df_alerts, weight_col="recency_weight")

        # Act
        result = baseline.predict_remaining("kyiv", elapsed_min=0.0)

        # Assert: the heavily-weighted long episode should dominate the median
        assert result["p50"] > 500.0
