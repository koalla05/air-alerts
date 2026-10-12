import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from lifelines import KaplanMeierFitter


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Weighted quantile via linear interpolation on the weighted CDF (Hazen's method).
    Degenerates to `np.percentile` when all weights are equal."""
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum_w = np.cumsum(w) - 0.5 * w
    cum_w /= np.sum(w)
    return float(np.interp(percentile / 100.0, cum_w, v))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or np.sum(weights) == 0:
        return 0.0
    return float(np.average(values, weights=weights))


class EmpiricalConditionalBaseline:
    """
    Non-parametric conditional empirical baseline (PLAN.md Section 5.1).
    Given region R and elapsed duration t:
      - Filters historical alerts where duration D > t.
      - Remaining time distribution: Y_rem = D - t.
      - Produces P10, P50 (median), P90 and P(ends within 15/30/60 min).
      - Gracefully falls back to global (national) empirical distribution if region sample is small.

    Optionally takes a `weight_col` (e.g. a recency weight) at fit time, so
    samples contribute unequally to the weighted percentiles/probabilities -
    see notebook 02 for a regime-shift comparison of this against training
    on a hard-cutoff recent-only window.
    """

    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
        self.region_durations: Dict[str, np.ndarray] = {}
        self.region_weights: Dict[str, np.ndarray] = {}
        self.global_durations: np.ndarray = np.array([])
        self.global_weights: np.ndarray = np.array([])

    def fit(
        self,
        df_alerts: pd.DataFrame,
        duration_col: str = "duration_min",
        weight_col: Optional[str] = None,
    ) -> "EmpiricalConditionalBaseline":
        valid = df_alerts.dropna(subset=[duration_col])
        self.global_durations = valid[duration_col].values
        self.global_weights = valid[weight_col].values if weight_col else np.ones(len(valid))

        self.region_durations = {}
        self.region_weights = {}
        for region_id, group in valid.groupby("region_id"):
            self.region_durations[region_id] = group[duration_col].values
            self.region_weights[region_id] = group[weight_col].values if weight_col else np.ones(len(group))
        return self

    def predict_remaining(
        self,
        region_id: str,
        elapsed_min: float,
        horizons_min: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if horizons_min is None:
            horizons_min = [15, 30, 60]

        durations = self.region_durations.get(region_id, self.global_durations)
        weights = self.region_weights.get(region_id, self.global_weights)
        mask = durations > elapsed_min
        surviving, surviving_w = durations[mask], weights[mask]

        if len(surviving) < self.min_samples:
            # Fallback to national
            mask_g = self.global_durations > elapsed_min
            surviving, surviving_w = self.global_durations[mask_g], self.global_weights[mask_g]

        if len(surviving) == 0:
            # If elapsed_min exceeds all historical alerts, return default conservative estimate
            return {
                "p10": 5.0,
                "p50": 15.0,
                "p90": 30.0,
                "probabilities": {f"within_{h}m": 0.5 for h in horizons_min},
                "sample_count": 0,
            }

        remaining = surviving - elapsed_min

        p10 = weighted_percentile(remaining, surviving_w, 10)
        p50 = weighted_percentile(remaining, surviving_w, 50)
        p90 = weighted_percentile(remaining, surviving_w, 90)

        probs = {}
        for h in horizons_min:
            probs[f"within_{h}m"] = weighted_mean((remaining <= h).astype(float), surviving_w)

        return {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "probabilities": probs,
            "sample_count": len(surviving),
        }

    def predict_batch(
        self,
        df_samples: pd.DataFrame,
        elapsed_col: str = "elapsed_min",
    ) -> pd.DataFrame:
        # Cache unique (region_id, elapsed) combinations for instantaneous vector lookup
        cache = {}
        unique_pairs = df_samples[["region_id", elapsed_col]].drop_duplicates().values
        for reg, t in unique_pairs:
            res = self.predict_remaining(reg, t)
            cache[(reg, t)] = (res["p10"], res["p50"], res["p90"])

        p10s, p50s, p90s = [], [], []
        for reg, t in zip(df_samples["region_id"].values, df_samples[elapsed_col].values):
            p10, p50, p90 = cache.get((reg, t), (5.0, 15.0, 30.0))
            p10s.append(p10)
            p50s.append(p50)
            p90s.append(p90)

        preds = pd.DataFrame(index=df_samples.index)
        preds["baseline_p10"] = p10s
        preds["baseline_p50"] = p50s
        preds["baseline_p90"] = p90s
        return preds


class KaplanMeierBaseline:
    """
    Survival Analysis baseline using Kaplan-Meier fitters per region (PLAN.md Section 5.1).
    Estimates survival curve S(t) = P(Duration > t).
    Conditional survival: S(Delta | t) = S(t + Delta) / S(t).
    Probability of alert ending within Delta: 1 - S(Delta | t).
    """

    def __init__(self):
        self.kmfs: Dict[str, KaplanMeierFitter] = {}
        self.global_kmf: Optional[KaplanMeierFitter] = None

    def fit(
        self,
        df_alerts: pd.DataFrame,
        duration_col: str = "duration_min",
        weight_col: Optional[str] = None,
    ) -> "KaplanMeierBaseline":
        weights = df_alerts[weight_col].values if weight_col else None

        self.global_kmf = KaplanMeierFitter()
        self.global_kmf.fit(df_alerts[duration_col], event_observed=np.ones(len(df_alerts)), weights=weights)

        self.kmfs = {}
        for region_id, group in df_alerts.groupby("region_id"):
            kmf = KaplanMeierFitter()
            group_weights = group[weight_col].values if weight_col else None
            kmf.fit(group[duration_col], event_observed=np.ones(len(group)), weights=group_weights)
            self.kmfs[region_id] = kmf
        return self

    def predict_survival_prob(
        self,
        region_id: str,
        elapsed_min: float,
        delta_min: float,
    ) -> float:
        """Computes conditional probability that alert ends within delta_min: 1 - S(t + delta) / S(t)."""
        kmf = self.kmfs.get(region_id, self.global_kmf)
        s_t = kmf.survival_function_at_times(elapsed_min).values[0]
        s_t_plus_delta = kmf.survival_function_at_times(elapsed_min + delta_min).values[0]

        if s_t <= 1e-6:
            return 1.0
        cond_surv = s_t_plus_delta / s_t
        return float(np.clip(1.0 - cond_surv, 0.0, 1.0))
