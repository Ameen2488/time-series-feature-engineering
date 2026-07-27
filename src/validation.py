"""
Time-series-aware validation utilities.

Wrappers around sklearn's TimeSeriesSplit plus custom walk-forward splitters
that enforce the forecast-horizon gap between train and test.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def make_walk_forward_splitter(
    n_splits: int = 5,
    max_train_size: int | None = None,
    test_size: int | None = None,
    forecast_horizon: int = 1,
) -> TimeSeriesSplit:
    """
    Construct a leakage-safe walk-forward cross-validator.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    max_train_size : int, optional
        Fixed training window size. If None, training window expands.
    test_size : int, optional
        Fixed test window size.
    forecast_horizon : int
        Gap between train and test to prevent leaking near-future data.
        Set this to your actual forecast horizon.

    Returns
    -------
    sklearn.model_selection.TimeSeriesSplit
    """
    return TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train_size,
        test_size=test_size,
        gap=max(0, forecast_horizon - 1),
    )


def describe_splits(
    splitter: TimeSeriesSplit,
    X: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """
    Return a summary DataFrame describing each CV fold — helpful for verifying
    that splits behave as expected before running expensive experiments.
    """
    records = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        train_start = train_idx[0]
        train_end = train_idx[-1]
        test_start = test_idx[0]
        test_end = test_idx[-1]

        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            records.append({
                "fold": fold,
                "train_start": X.index[train_start],
                "train_end": X.index[train_end],
                "test_start": X.index[test_start],
                "test_end": X.index[test_end],
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "gap": test_start - train_end - 1,
            })
        else:
            records.append({
                "fold": fold,
                "train_start_idx": train_start,
                "train_end_idx": train_end,
                "test_start_idx": test_start,
                "test_end_idx": test_end,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "gap": test_start - train_end - 1,
            })

    return pd.DataFrame(records)


def rolling_origin_splits(
    n_samples: int,
    initial_train_size: int,
    test_size: int = 1,
    step: int = 1,
    forecast_horizon: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate rolling-origin (expanding-window) splits by hand — useful when
    sklearn's TimeSeriesSplit doesn't fit the exact validation pattern.

    Yields (train_indices, test_indices) tuples.
    """
    origin = initial_train_size
    while origin + forecast_horizon + test_size - 1 < n_samples:
        train_idx = np.arange(0, origin)
        test_start = origin + forecast_horizon - 1
        test_end = test_start + test_size
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx
        origin += step
