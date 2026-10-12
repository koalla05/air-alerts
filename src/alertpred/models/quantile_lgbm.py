import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from ..config import (
    QUANTILES,
    TASK_A_HORIZONS_MIN,
    TASK_B_HORIZONS_MIN,
    MODELS_DIR,
)


class LightGBMQuantileModel:
    """
    Gradient boosted quantile regressors (alpha = 0.1, 0.5, 0.9) to estimate
    median and uncertainty intervals [P10, P90] for remaining duration.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ):
        self.quantiles = quantiles or QUANTILES
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols or ["region_id"]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.models: Dict[float, lgb.LGBMRegressor] = {}

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def fit(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        weight_col: Optional[str] = None,
    ) -> "LightGBMQuantileModel":
        if self.feature_cols is None:
            exclude = {
                "as_of_ts", "start_ts", "gap_start_ts", "total_duration_min",
                "total_gap_min", "region_en", target_col,
            }
            if weight_col:
                exclude.add(weight_col)
            target_prefixes = ("target_",)
            self.feature_cols = [
                c for c in df_train.columns
                if c not in exclude and not c.startswith(target_prefixes)
            ]

        X_train = self._prepare_data(df_train)
        y_train = df_train[target_col].values
        sample_weight = df_train[weight_col].values if weight_col else None

        self.models = {}
        for alpha in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=42,
                verbosity=-1,
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)
            self.models[alpha] = model

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_data(df)
        preds = pd.DataFrame(index=df.index)
        for alpha, model in self.models.items():
            col_name = f"pred_p{int(alpha * 100)}"
            # Clip negative predictions to 0 (remaining time cannot be negative)
            preds[col_name] = np.maximum(0.0, model.predict(X))

        # Enforce quantile monotonicity (P10 <= P50 <= P90)
        p10_col = f"pred_p{int(self.quantiles[0] * 100)}"
        p50_col = f"pred_p{int(self.quantiles[1] * 100)}"
        p90_col = f"pred_p{int(self.quantiles[2] * 100)}"

        if p10_col in preds and p50_col in preds and p90_col in preds:
            preds[p50_col] = np.maximum(preds[p10_col], preds[p50_col])
            preds[p90_col] = np.maximum(preds[p50_col], preds[p90_col])

        return preds


class LightGBMProbabilityModel:
    """
    LightGBM binary classifiers predicting threshold probabilities:
    e.g. P(ends within 15 / 30 / 60 min) or P(starts within 1 / 3 / 6 h).
    """

    def __init__(
        self,
        target_horizons: List[int],
        horizon_prefix: str = "ends_within_",
        horizon_suffix: str = "m",
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
    ):
        self.target_horizons = target_horizons
        self.horizon_prefix = horizon_prefix
        self.horizon_suffix = horizon_suffix
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols or ["region_id"]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models: Dict[int, lgb.LGBMClassifier] = {}

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def fit(self, df_train: pd.DataFrame, weight_col: Optional[str] = None) -> "LightGBMProbabilityModel":
        X_train = self._prepare_data(df_train)
        sample_weight = df_train[weight_col].values if weight_col else None
        self.models = {}

        for h in self.target_horizons:
            target_col = f"target_{self.horizon_prefix}{h}{self.horizon_suffix}"
            if target_col not in df_train.columns:
                continue

            y_train = df_train[target_col].values
            clf = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=42,
                verbosity=-1,
            )
            clf.fit(X_train, y_train, sample_weight=sample_weight)
            self.models[h] = clf

        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_data(df)
        probs = pd.DataFrame(index=df.index)
        for h, clf in self.models.items():
            probs[f"prob_{self.horizon_prefix}{h}{self.horizon_suffix}"] = clf.predict_proba(X)[:, 1]
        return probs


class AlertPredictionPipeline:
    """
    End-to-End Orchestrator for training and inference across Task A and Task B.
    """

    def __init__(self):
        self.task_a_quantile = LightGBMQuantileModel(
            quantiles=[0.10, 0.50, 0.90],
            feature_cols=[
                "region_id", "elapsed_min", "hour_sin", "hour_cos",
                "dow_sin", "dow_cos", "is_weekend", "month",
            ],
        )
        self.task_a_probs = LightGBMProbabilityModel(
            target_horizons=TASK_A_HORIZONS_MIN,
            horizon_prefix="ends_within_",
            horizon_suffix="m",
            feature_cols=[
                "region_id", "elapsed_min", "hour_sin", "hour_cos",
                "dow_sin", "dow_cos", "is_weekend", "month",
            ],
        )
        self.task_b_quantile = LightGBMQuantileModel(
            quantiles=[0.10, 0.50, 0.90],
            feature_cols=[
                "region_id", "elapsed_quiet_min", "prev_alert_duration_min",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month",
            ],
        )
        self.task_b_probs = LightGBMProbabilityModel(
            target_horizons=[1, 3, 6],
            horizon_prefix="starts_within_",
            horizon_suffix="h",
            feature_cols=[
                "region_id", "elapsed_quiet_min", "prev_alert_duration_min",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month",
            ],
        )

    def fit_task_a(self, df_train_a: pd.DataFrame, weight_col: Optional[str] = None):
        print("Training Task A (Remaining alert duration quantiles P10/P50/P90)...")
        self.task_a_quantile.fit(df_train_a, target_col="target_remaining_min", weight_col=weight_col)
        print("Training Task A (Probability of alert end within 15/30/60m)...")
        self.task_a_probs.fit(df_train_a, weight_col=weight_col)
        return self

    def fit_task_b(self, df_train_b: pd.DataFrame, weight_col: Optional[str] = None):
        print("Training Task B (Quiet gap until next alert quantiles P10/P50/P90)...")
        self.task_b_quantile.fit(df_train_b, target_col="target_remaining_gap_min", weight_col=weight_col)
        print("Training Task B (Probability of next alert within 1h/3h/6h)...")
        self.task_b_probs.fit(df_train_b, weight_col=weight_col)
        return self

    def predict_active_alert(
        self,
        region_id: str,
        elapsed_min: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """Inference for an active alert in a specific region."""
        ts = timestamp or pd.Timestamp.now(tz="UTC")
        hour = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek
        
        sample = pd.DataFrame([{
            "region_id": region_id,
            "elapsed_min": float(elapsed_min),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "is_weekend": int(dow >= 5),
            "month": ts.month,
        }])

        preds_quant = self.task_a_quantile.predict(sample).iloc[0].to_dict()
        preds_prob = self.task_a_probs.predict_proba(sample).iloc[0].to_dict()

        return {
            "task": "A_active_alert",
            "region_id": region_id,
            "elapsed_min": elapsed_min,
            "predicted_remaining_median_min": round(preds_quant["pred_p50"], 1),
            "predicted_range_p10_p90_min": [
                round(preds_quant["pred_p10"], 1),
                round(preds_quant["pred_p90"], 1),
            ],
            "probabilities": {
                k: round(float(v), 3) for k, v in preds_prob.items()
            },
        }

    def predict_quiet_region(
        self,
        region_id: str,
        elapsed_quiet_min: float,
        prev_alert_duration_min: float = 45.0,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """Inference for a quiet region predicting time until the next alert."""
        ts = timestamp or pd.Timestamp.now(tz="UTC")
        hour = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek
        
        sample = pd.DataFrame([{
            "region_id": region_id,
            "elapsed_quiet_min": float(elapsed_quiet_min),
            "prev_alert_duration_min": float(prev_alert_duration_min),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "is_weekend": int(dow >= 5),
            "month": ts.month,
        }])

        preds_quant = self.task_b_quantile.predict(sample).iloc[0].to_dict()
        preds_prob = self.task_b_probs.predict_proba(sample).iloc[0].to_dict()

        return {
            "task": "B_quiet_period",
            "region_id": region_id,
            "elapsed_quiet_min": elapsed_quiet_min,
            "predicted_next_alert_gap_median_min": round(preds_quant["pred_p50"], 1),
            "predicted_range_p10_p90_min": [
                round(preds_quant["pred_p10"], 1),
                round(preds_quant["pred_p90"], 1),
            ],
            "probabilities": {
                k: round(float(v), 3) for k, v in preds_prob.items()
            },
        }

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        save_path = Path(filepath) if filepath else MODELS_DIR / "alertpred_pipeline.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        print(f"Saved pipeline artifacts to: {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "AlertPredictionPipeline":
        load_path = Path(filepath) if filepath else MODELS_DIR / "alertpred_pipeline.joblib"
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at: {load_path}")
        return joblib.load(load_path)
