"""Metrics shared by the multi-horizon SSE forecast evaluations."""

from __future__ import annotations

import numpy as np


def anomaly_correlation_np(
    pred: np.ndarray,
    true: np.ndarray,
    climatology: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1.0e-12,
) -> float:
    """Return an equal-weight ACC relative to training-period climatology.

    ``climatology`` is the temporal mean field of the training split. The
    anomalies are not re-centred over the test samples, which distinguishes ACC
    from the existing Pearson correlation coefficient. ACC is evaluated over
    the spatial dimension for each forecast sample and then averaged equally
    over samples. Every selected spatial point contributes equally; ``mask``
    only selects the evaluation domain.
    """
    pred_anomaly = np.asarray(pred, dtype=np.float64) - np.asarray(
        climatology, dtype=np.float64
    )
    true_anomaly = np.asarray(true, dtype=np.float64) - np.asarray(
        climatology, dtype=np.float64
    )
    if pred_anomaly.shape != true_anomaly.shape:
        raise ValueError(
            f"pred and true must have the same shape, got "
            f"{pred_anomaly.shape} and {true_anomaly.shape}"
        )
    if pred_anomaly.ndim == 1:
        pred_anomaly = pred_anomaly[None, :]
        true_anomaly = true_anomaly[None, :]
    else:
        pred_anomaly = pred_anomaly.reshape(pred_anomaly.shape[0], -1)
        true_anomaly = true_anomaly.reshape(true_anomaly.shape[0], -1)

    if mask is None:
        weights = np.ones_like(true_anomaly, dtype=np.float64)
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.ndim == 1:
            selected = selected[None, :]
        else:
            selected = selected.reshape(selected.shape[0], -1)
        if selected.shape != true_anomaly.shape:
            raise ValueError(
                f"mask must match pred and true, got {selected.shape} and "
                f"{true_anomaly.shape}"
            )
        weights = selected.astype(np.float64)

    numerator = np.sum(weights * pred_anomaly * true_anomaly, axis=1)
    pred_energy = np.sum(weights * pred_anomaly**2, axis=1)
    true_energy = np.sum(weights * true_anomaly**2, axis=1)
    denominator = np.sqrt(pred_energy * true_energy)
    valid = denominator > eps
    if not np.any(valid):
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))

