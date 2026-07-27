"""
Feature engineering utilities for time series.

All feature-generating functions here are **leakage-safe by construction**:
they use .shift() before any rolling or expanding operation, so a feature
computed at time t never sees the value of the target at time t.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Sequence[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add lag features of `target_col` to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    target_col : str
        Column to lag.
    lags : sequence of int
        Positive integers indicating lag periods.
    prefix : str, optional
        Column name prefix. Defaults to f"{target_col}_lag".

    Returns
    -------
    pd.DataFrame with new lag columns appended.
    """
    df = df.copy()
    prefix = prefix or f"{target_col}_lag"
    for lag in lags:
        if lag <= 0:
            raise ValueError(f"Lags must be positive integers, got {lag}.")
        df[f"{prefix}_{lag}"] = df[target_col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Window features (rolling & expanding statistics)
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: Sequence[int],
    functions: Sequence[str] = ("mean", "std"),
    min_horizon: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add rolling window features of `target_col`, leakage-safe.

    The `min_horizon` parameter shifts the series by that many periods before
    applying the rolling window. Set `min_horizon` to your forecast horizon
    to guarantee no leakage in multi-step forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    target_col : str
        Column to compute rolling features on.
    windows : sequence of int
        Window sizes in periods.
    functions : sequence of str
        Names of pandas rolling reductions ('mean', 'std', 'min', 'max',
        'median', 'sum', 'skew', 'kurt').
    min_horizon : int
        Number of periods to shift before rolling. Must be >= 1.
    prefix : str, optional
        Column name prefix. Defaults to f"{target_col}_roll".

    Returns
    -------
    pd.DataFrame with new rolling feature columns appended.
    """
    if min_horizon < 1:
        raise ValueError(
            "min_horizon must be >= 1 to avoid leaking the current target."
        )

    df = df.copy()
    prefix = prefix or f"{target_col}_roll"
    shifted = df[target_col].shift(min_horizon)

    for window in windows:
        rolling = shifted.rolling(window=window, min_periods=1)
        for func in functions:
            col = f"{prefix}_{func}_{window}"
            df[col] = getattr(rolling, func)()

    return df


def add_expanding_features(
    df: pd.DataFrame,
    target_col: str,
    functions: Sequence[str] = ("mean", "std"),
    min_horizon: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add expanding window features (cumulative statistics over all past values).
    """
    if min_horizon < 1:
        raise ValueError("min_horizon must be >= 1.")

    df = df.copy()
    prefix = prefix or f"{target_col}_expand"
    shifted = df[target_col].shift(min_horizon)

    for func in functions:
        col = f"{prefix}_{func}"
        df[col] = getattr(shifted.expanding(min_periods=1), func)()

    return df


def add_ewm_features(
    df: pd.DataFrame,
    target_col: str,
    spans: Sequence[int],
    min_horizon: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add exponentially-weighted moving average features.
    """
    if min_horizon < 1:
        raise ValueError("min_horizon must be >= 1.")

    df = df.copy()
    prefix = prefix or f"{target_col}_ewm"
    shifted = df[target_col].shift(min_horizon)

    for span in spans:
        df[f"{prefix}_span_{span}"] = shifted.ewm(span=span, adjust=False).mean()

    return df


# ---------------------------------------------------------------------------
# Datetime features
# ---------------------------------------------------------------------------

def add_datetime_features(
    df: pd.DataFrame,
    date_col: str | None = None,
    features: Iterable[str] = (
        "dayofweek", "day", "month", "quarter", "year",
        "weekofyear", "dayofyear", "is_weekend",
    ),
) -> pd.DataFrame:
    """
    Extract calendar features from a datetime column or DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str, optional
        Column name with datetime values. If None, uses the DataFrame's index
        (which must be a DatetimeIndex).
    features : iterable of str
        Which datetime attributes to extract. Supported:
        'dayofweek', 'day', 'month', 'quarter', 'year', 'weekofyear',
        'dayofyear', 'hour', 'minute', 'is_weekend', 'is_month_start',
        'is_month_end', 'is_quarter_start', 'is_quarter_end'.

    Returns
    -------
    pd.DataFrame with datetime feature columns appended.
    """
    df = df.copy()
    dt = df[date_col] if date_col else df.index

    if not isinstance(dt, (pd.DatetimeIndex, pd.Series)):
        raise TypeError("date_col or index must contain datetime values.")

    dt_accessor = dt.dt if isinstance(dt, pd.Series) else dt

    mapping = {
        "dayofweek": lambda x: x.dayofweek,
        "day": lambda x: x.day,
        "month": lambda x: x.month,
        "quarter": lambda x: x.quarter,
        "year": lambda x: x.year,
        "weekofyear": lambda x: x.isocalendar().week.astype(int)
            if hasattr(x, "isocalendar") else pd.Int64Index(x.isocalendar().week),
        "dayofyear": lambda x: x.dayofyear,
        "hour": lambda x: x.hour,
        "minute": lambda x: x.minute,
        "is_weekend": lambda x: (x.dayofweek >= 5).astype(int),
        "is_month_start": lambda x: x.is_month_start.astype(int),
        "is_month_end": lambda x: x.is_month_end.astype(int),
        "is_quarter_start": lambda x: x.is_quarter_start.astype(int),
        "is_quarter_end": lambda x: x.is_quarter_end.astype(int),
    }

    for feat in features:
        if feat not in mapping:
            raise ValueError(f"Unsupported datetime feature: {feat}")
        df[feat] = mapping[feat](dt_accessor)

    return df


# ---------------------------------------------------------------------------
# Cyclical encoding
# ---------------------------------------------------------------------------

def add_cyclical_encoding(
    df: pd.DataFrame,
    col: str,
    period: int,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Encode a cyclical feature (e.g. day-of-week, month) as sin/cos pair.

    This preserves the cyclical relationship — the model learns that
    December (12) and January (1) are adjacent, not maximally distant.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to encode.
    period : int
        The period of the cycle (7 for day-of-week, 12 for month, 24 for hour).
    drop_original : bool
        If True, drop the original numeric column.
    """
    df = df.copy()
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    if drop_original:
        df = df.drop(columns=[col])

    return df


# ---------------------------------------------------------------------------
# Fourier terms (for long/multiple seasonalities)
# ---------------------------------------------------------------------------

def add_fourier_terms(
    df: pd.DataFrame,
    period: float,
    n_terms: int,
    prefix: str = "fourier",
) -> pd.DataFrame:
    """
    Add Fourier terms indexed by a synthetic time column starting at 0.

    Useful for encoding long or multiple seasonalities (e.g. yearly
    seasonality on daily data, where a dummy per day-of-year is too sparse).

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    period : float
        Length of the seasonal period in observations
        (e.g. 365.25 for yearly on daily data, 52 for yearly on weekly data).
    n_terms : int
        Number of Fourier pairs (higher = more flexible).
    prefix : str
        Column name prefix.
    """
    df = df.copy()
    t = np.arange(len(df))
    for k in range(1, n_terms + 1):
        df[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


# ---------------------------------------------------------------------------
# Convenience: build the full leakage-safe feature matrix
# ---------------------------------------------------------------------------

def build_forecasting_feature_matrix(
    series: pd.Series,
    lags: Sequence[int] = (1, 7, 14, 28),
    rolling_windows: Sequence[int] = (7, 14, 28),
    rolling_functions: Sequence[str] = ("mean", "std"),
    forecast_horizon: int = 1,
    include_datetime: bool = True,
) -> pd.DataFrame:
    """
    Build a complete, leakage-safe feature matrix for time series forecasting.

    Convenience wrapper combining lag, rolling, and datetime features.
    All rolling and lag operations respect the given forecast_horizon.

    Parameters
    ----------
    series : pd.Series
        Target series with DatetimeIndex.
    lags : sequence of int
        Lag periods. All lags must be >= forecast_horizon to be usable.
    rolling_windows : sequence of int
    rolling_functions : sequence of str
    forecast_horizon : int
        Minimum shift applied before any rolling / lag operation.
    include_datetime : bool

    Returns
    -------
    pd.DataFrame with 'target' column and all features. Rows with NaN dropped.
    """
    if series.name is None:
        series = series.rename("target")
    target_col = "target"

    df = series.rename(target_col).to_frame()

    # Lag features (all lags must be >= horizon to be known at forecast time)
    valid_lags = [lag for lag in lags if lag >= forecast_horizon]
    if len(valid_lags) < len(lags):
        dropped = [lag for lag in lags if lag < forecast_horizon]
        print(f"[warn] Dropped lags {dropped} — smaller than forecast horizon.")
    df = add_lag_features(df, target_col, valid_lags)

    # Rolling features
    df = add_rolling_features(
        df, target_col, rolling_windows, rolling_functions,
        min_horizon=forecast_horizon,
    )

    # Datetime features
    if include_datetime:
        df = add_datetime_features(df)
        df = add_cyclical_encoding(df, "dayofweek", period=7, drop_original=False)
        df = add_cyclical_encoding(df, "month", period=12, drop_original=False)

    return df.dropna()
