"""Feature engineering for real-time air alert duration and inter-alert gap prediction."""

from .builder import (
    sample_as_of_alert_moments,
    sample_as_of_gap_moments,
    extract_point_in_time_features,
)

__all__ = [
    "sample_as_of_alert_moments",
    "sample_as_of_gap_moments",
    "extract_point_in_time_features",
]

