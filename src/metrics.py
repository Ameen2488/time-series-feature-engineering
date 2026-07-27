"""
Forecasting metrics.

Includes the standard error metrics plus per-horizon evaluation and
Vandeput's Score = MAE + |Bias|, which balances accuracy with unbiasedness
for demand forecasting use cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean forecast bias (positive = over-forecast on average)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_pred - y_true))


def wmape(y_true, y_pred) -> float:
    """
    Weighted Mean Absolute Percentage Error = sum(|error|) / sum(|actual|).

    Mathematically equivalent to MAE / mean(actual). More robust than MAPE
    for series with near-zero actuals.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mase(y_true, y_pred, y_train, seasonality: int = 1) -> float:
    """
    Mean Absolute Scaled Error — scale-free, comparable across series.

    Scales by the MAE of a seasonal naive forecast on the training data.
    """
    y_true, y_pred, y_train = (
        np.asarray(y_true), np.asarray(y_pred), np.asarray(y_train)
    )
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def vandeput_score(y_true, y_pred) -> float:
    """
    Vandeput's Score = MAE + |Bias|.

    Balances accuracy (MAE) with unbiasedness (|Bias|). A model with low MAE
    but systematic bias scores worse than a slightly less accurate but
    unbiased model — which matches business needs in demand forecasting.
    """
    return mae(y_true, y_pred) + abs(bias(y_true, y_pred))


def evaluate_per_horizon(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
    horizon_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute error metrics per forecast horizon.

    Parameters
    ----------
    y_true : (n_samples, n_horizons) array-like
    y_pred : (n_samples, n_horizons) array-like
    horizon_names : list of str, optional
        Column names for horizons. Defaults to h_1, h_2, ...

    Returns
    -------
    pd.DataFrame with columns ['horizon', 'mae', 'rmse', 'bias', 'wmape',
    'vandeput_score'].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
        )

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_horizons = y_true.shape[1]
    if horizon_names is None:
        horizon_names = [f"h_{i+1}" for i in range(n_horizons)]

    records = []
    for i, name in enumerate(horizon_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        records.append({
            "horizon": name,
            "mae": mae(yt, yp),
            "rmse": rmse(yt, yp),
            "bias": bias(yt, yp),
            "wmape": wmape(yt, yp),
            "vandeput_score": vandeput_score(yt, yp),
        })

    return pd.DataFrame(records)
