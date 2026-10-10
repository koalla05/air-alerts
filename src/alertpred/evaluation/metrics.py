import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss for quantile alpha."""
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1.0) * err)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Calculates empirical coverage: fraction of true values falling between lower and upper bounds."""
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def evaluate_point_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE and Median Absolute Error in minutes."""
    abs_err = np.abs(y_true - y_pred)
    return {
        "mae_min": float(np.mean(abs_err)),
        "median_ae_min": float(np.median(abs_err)),
    }


def evaluate_quantile_predictions(
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> Dict[str, float]:
    """Evaluates median MAE, pinball losses, and P10-P90 interval coverage."""
    pt_metrics = evaluate_point_predictions(y_true, p50)
    loss_p10 = pinball_loss(y_true, p10, 0.10)
    loss_p50 = pinball_loss(y_true, p50, 0.50)
    loss_p90 = pinball_loss(y_true, p90, 0.90)
    cov_80 = interval_coverage(y_true, p10, p90)

    return {
        "mae_p50_min": pt_metrics["mae_min"],
        "median_ae_p50_min": pt_metrics["median_ae_min"],
        "pinball_loss_p10": loss_p10,
        "pinball_loss_p50": loss_p50,
        "pinball_loss_p90": loss_p90,
        "mean_pinball_loss": float(np.mean([loss_p10, loss_p50, loss_p90])),
        "coverage_p10_p90": cov_80,
    }


def evaluate_classification_probabilities(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Evaluates probability calibration, Brier score, and accuracy at 0.5 threshold."""
    # Ensure binary targets have at least both classes present for AUC
    res = {
        "brier_score": float(brier_score_loss(y_true_binary, y_prob)),
        "accuracy": float(accuracy_score(y_true_binary, y_prob >= 0.5)),
    }
    if len(np.unique(y_true_binary)) > 1:
        res["roc_auc"] = float(roc_auc_score(y_true_binary, y_prob))
        res["log_loss"] = float(log_loss(y_true_binary, y_prob, labels=[0, 1]))
    return res

