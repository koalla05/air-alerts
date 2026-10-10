import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from lifelines import KaplanMeierFitter


class EmpiricalConditionalBaseline:
    """
    Non-parametric conditional empirical baseline (PLAN.md Section 5.1).
    Given region R and elapsed duration t:
      - Filters historical alerts where duration D > t.
      - Remaining time distribution: Y_rem = D - t.
      - Produces P10, P50 (median), P90 and P(ends within 15/30/60 min).
      - Gracefully falls back to global (national) empirical distribution if region sample is small.
    """

    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
        self.region_durations: Dict[str, np.ndarray] = {}
        self.global_durations: np.ndarray = np.array([])

    def fit(self, df_alerts: pd.DataFrame, duration_col: str = "duration_min") -> "EmpiricalConditionalBaseline":
        self.global_durations = df_alerts[duration_col].dropna().values
        self.region_durations = {}
        for region_id, group in df_alerts.groupby("region_id"):
            self.region_durations[region_id] = group[duration_col].dropna().values
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
        # Filter alerts that lasted longer than elapsed_min
        surviving = durations[durations > elapsed_min]

        if len(surviving) < self.min_samples:
            # Fallback to national
            surviving = self.global_durations[self.global_durations > elapsed_min]

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

        p10 = float(np.percentile(remaining, 10))
        p50 = float(np.percentile(remaining, 50))
        p90 = float(np.percentile(remaining, 90))

        probs = {}
        for h in horizons_min:
            probs[f"within_{h}m"] = float(np.mean(remaining <= h))

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

    def fit(self, df_alerts: pd.DataFrame, duration_col: str = "duration_min") -> "KaplanMeierBaseline":
        self.global_kmf = KaplanMeierFitter()
        self.global_kmf.fit(df_alerts[duration_col], event_observed=np.ones(len(df_alerts)))

        self.kmfs = {}
        for region_id, group in df_alerts.groupby("region_id"):
            kmf = KaplanMeierFitter()
            kmf.fit(group[duration_col], event_observed=np.ones(len(group)))
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
