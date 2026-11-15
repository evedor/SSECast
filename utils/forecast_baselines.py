"""Causal, non-neural reference forecasts for multi-horizon SSE evaluation.

Every baseline uses observations available at a forecast origin only.  The AR
coefficient and recurrence lag are estimated from the training split supplied
as ``history_series``; they therefore do not use test-period information.
"""

from __future__ import annotations

import numpy as np


def _ar1_coefficient(history_release: np.ndarray) -> float:
    """Estimate one stable AR(1) coefficient for daily release increments."""
    increments = np.diff(history_release.astype(np.float64, copy=False), axis=0)
    if increments.shape[0] < 2:
        return 0.0
    numerator = float(np.sum(increments[1:] * increments[:-1]))
    denominator = float(np.sum(increments[:-1] ** 2))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.clip(numerator / denominator, -0.95, 0.95))


def _empirical_recurrence_lag(history_release: np.ndarray, horizon: int) -> int:
    """Find a long-lag peak in training release-increment autocorrelation."""
    activity = np.linalg.norm(np.diff(history_release, axis=0), axis=1).astype(np.float64)
    if activity.size < 2 * max(2 * horizon, 30):
        return max(horizon, 1)

    centered = activity - np.mean(activity)
    min_lag = max(2 * int(horizon), 30)
    max_lag = min(730, centered.size // 2)
    if max_lag < min_lag:
        return min_lag

    scores = []
    for lag in range(min_lag, max_lag + 1):
        earlier = centered[:-lag]
        later = centered[lag:]
        scale = np.linalg.norm(earlier) * np.linalg.norm(later)
        scores.append(float(np.dot(earlier, later) / scale) if scale > 0.0 else -np.inf)
    return int(min_lag + int(np.argmax(scores)))


def build_noa_global_geodesy(
    history_series: np.ndarray,
    test_series: np.ndarray,
    horizon: int,
    neighbor_fraction: float = 0.01,
    query_batch_size: int = 32,
) -> np.ndarray:
    """Forecast fields with a causal nearest-observed-analogue (NOA) model.

    The archive contains only observations before the test split. Candidate
    states are ranked by Euclidean distance over the complete normalized
    geodetic source field. The closest ``neighbor_fraction`` of states provide
    inverse-distance-weighted future increments, which are added to the current
    test state. This increment formulation preserves the forecast initial
    condition while retaining the observed evolution of the analogues.
    """
    horizon = int(horizon)
    fraction = float(neighbor_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("neighbor_fraction must lie in (0, 1].")

    history = np.asarray(history_series[:, 0, :], dtype=np.float32)
    test = np.asarray(test_series[:, 0, :], dtype=np.float32)
    n_query = test.shape[0] - horizon - 1
    n_candidate = history.shape[0] - horizon
    if n_query <= 0 or n_candidate <= 0:
        raise ValueError("History or test series is too short for the requested horizon.")

    candidates = history[:n_candidate]
    queries = test[1 : 1 + n_query]
    neighbor_count = max(1, int(np.ceil(fraction * n_candidate)))
    candidate_norm = np.sum(candidates.astype(np.float64) ** 2, axis=1)
    forecasts = np.empty((n_query, horizon, history.shape[1]), dtype=np.float32)
    lead_offsets = np.arange(1, horizon + 1, dtype=np.int64)

    for start in range(0, n_query, int(query_batch_size)):
        stop = min(start + int(query_batch_size), n_query)
        query_batch = queries[start:stop]
        query_norm = np.sum(query_batch.astype(np.float64) ** 2, axis=1)[:, None]
        distances2 = query_norm + candidate_norm[None, :]
        distances2 -= 2.0 * (
            query_batch.astype(np.float64) @ candidates.astype(np.float64).T
        )
        np.maximum(distances2, 0.0, out=distances2)

        selected = np.argpartition(
            distances2, kth=neighbor_count - 1, axis=1
        )[:, :neighbor_count]
        selected_distances = np.sqrt(
            np.take_along_axis(distances2, selected, axis=1)
        )
        weights = 1.0 / np.maximum(selected_distances, 1.0e-8)
        weights /= np.sum(weights, axis=1, keepdims=True)

        for local_index, query_index in enumerate(range(start, stop)):
            analog_origins = selected[local_index]
            analog_future = history[analog_origins[:, None] + lead_offsets[None, :]]
            analog_increment = analog_future - history[analog_origins, None, :]
            weighted_increment = np.einsum(
                "k,khs->hs",
                weights[local_index],
                analog_increment,
                optimize=True,
            )
            forecasts[query_index] = queries[query_index] + weighted_increment

    return forecasts


def build_baselines(
    test_series: np.ndarray,
    horizon: int,
    history_series: np.ndarray,
    analog_history_series: np.ndarray | None = None,
    noa_neighbor_fraction: float = 0.01,
) -> dict[str, np.ndarray]:
    """Return causal statistical forecasts, including global-geodesy NOA.

    ``test_series`` and ``history_series`` are normalized field sequences with
    dimensions ``time x channel x space``.  Only the release channel is used,
    matching the existing metric calculation.
    """
    horizon = int(horizon)
    release = np.asarray(test_series[:, 0, :], dtype=np.float32)
    history_release = np.asarray(history_series[:, 0, :], dtype=np.float32)
    n = release.shape[0] - horizon - 1
    if n <= 0:
        raise ValueError(f"Series length {release.shape[0]} is too short for horizon={horizon}.")

    base = release[1 : 1 + n]
    previous = release[:n]
    increment = base - previous
    leads = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]

    forecasts: dict[str, np.ndarray] = {
        "persistence": np.repeat(base[:, None, :], horizon, axis=1),
        "linear extrapolation": base[:, None, :] + increment[:, None, :] * leads,
    }

    # Existing alpha-beta local-trend filter, retained under its established name.
    alpha, beta = 0.35, 0.08
    level = np.zeros_like(release)
    trend = np.zeros_like(release)
    level[0] = release[0]
    trend[0] = release[1] - release[0]
    for index in range(1, release.shape[0]):
        one_step = level[index - 1] + trend[index - 1]
        residual = release[index] - one_step
        level[index] = one_step + alpha * residual
        trend[index] = trend[index - 1] + beta * residual
    forecasts["Kalman local trend"] = level[1 : 1 + n, None, :] + leads * trend[1 : 1 + n, None, :]

    # AR(1) model for the daily release increment, calibrated on training only.
    phi = _ar1_coefficient(history_release)
    if abs(phi - 1.0) < 1e-8:
        ar_increment_sum = increment[:, None, :] * leads
    else:
        lead_vector = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]
        ar_increment_sum = increment[:, None, :] * (
            phi * (1.0 - phi**lead_vector) / (1.0 - phi)
        )
    forecasts["AR(1) increment"] = base[:, None, :] + ar_increment_sum

    # Replay the release increments from one empirically inferred recurrence lag
    # earlier.  The lag is at least twice the forecast horizon, so the replayed
    # segment always lies in the causal past of the forecast origin.
    recurrence_lag = _empirical_recurrence_lag(history_release, horizon)
    combined = np.concatenate([history_release, release], axis=0)
    origins = history_release.shape[0] + np.arange(1, n + 1)
    source = origins - recurrence_lag
    recurrence = forecasts["persistence"].copy()
    valid = source >= 0
    if np.any(valid):
        source_valid = source[valid]
        source_steps = source_valid[:, None] + np.arange(1, horizon + 1)[None, :]
        recurrence[valid] = base[valid, None, :] + (
            combined[source_steps] - combined[source_valid[:, None]]
        )
    forecasts["empirical recurrence"] = recurrence
    analog_archive = history_series if analog_history_series is None else analog_history_series
    forecasts["NOA global geodesy"] = build_noa_global_geodesy(
        analog_archive,
        test_series,
        horizon,
        neighbor_fraction=noa_neighbor_fraction,
    )
    return forecasts
