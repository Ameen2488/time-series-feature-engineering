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


def add_lag_features_horizon_aware(
    df: pd.DataFrame,
    target_col: str,
    lags: Sequence[int],
    forecast_horizon: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add lag features, dropping any lag smaller than the forecast horizon.

    Guarantees no temporal leakage in multi-step forecasting scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    target_col : str
        Column to lag.
    lags : sequence of int
        Candidate lag periods.
    forecast_horizon : int, default=1
        Forecast horizon H. Any lag < H will be dropped with a warning.
    prefix : str, optional
        Column name prefix. Defaults to f"{target_col}_lag".

    Returns
    -------
    pd.DataFrame with valid horizon-aware lag columns appended.
    """
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be >= 1.")

    valid_lags = [lag for lag in lags if lag >= forecast_horizon]
    if len(valid_lags) < len(lags):
        dropped = [lag for lag in lags if lag < forecast_horizon]
        import warnings
        warnings.warn(
            f"Dropped lags {dropped} — smaller than forecast horizon ({forecast_horizon}).",
            UserWarning,
            stacklevel=2,
        )

    return add_lag_features(df, target_col=target_col, lags=valid_lags, prefix=prefix)


def cross_correlation(
    target: pd.Series,
    exog: pd.Series,
    max_lag: int = 20,
) -> tuple[list[int], list[float]]:
    """
    Compute cross-correlation function (CCF) between target and exogenous variable.

    Positive lag indicates `exog` leads `target` by `k` periods: corr(target_t, exog_{t-k}).
    Negative lag indicates `target` leads `exog` by `k` periods.

    Parameters
    ----------
    target : pd.Series
        Target time series.
    exog : pd.Series
        Exogenous time series.
    max_lag : int, default=20
        Maximum lag to evaluate in both directions.

    Returns
    -------
    tuple of (lags, correlations)
        lags : list of int from -max_lag to +max_lag
        correlations : list of float Pearson correlations
    """
    ccf_vals = []
    lags = list(range(-max_lag, max_lag + 1))
    for lag in lags:
        corr = target.corr(exog.shift(lag))
        ccf_vals.append(float(corr) if not pd.isna(corr) else 0.0)

    return lags, ccf_vals




# ---------------------------------------------------------------------------
# Window features (rolling & expanding statistics)
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: Sequence[int],
    functions: Sequence[str] = ("mean", "std"),
    min_horizon: int = 1,
    min_periods: int | None = None,
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
    min_periods : int, optional
        Minimum number of observations in window required to have a value.
        Defaults to 1.
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
    mp = min_periods if min_periods is not None else 1

    for window in windows:
        rolling = shifted.rolling(window=window, min_periods=mp)
        for func in functions:
            col = f"{prefix}_{func}_{window}"
            df[col] = getattr(rolling, func)()

    return df


def add_expanding_features(
    df: pd.DataFrame,
    target_col: str,
    functions: Sequence[str] = ("mean", "std"),
    min_periods: int = 1,
    min_horizon: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add expanding window features (cumulative statistics over all past values).

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    target_col : str
        Column to compute expanding features on.
    functions : sequence of str
        Names of pandas rolling/expanding reductions ('mean', 'std', 'min', 'max', etc.).
    min_periods : int, default=1
        Minimum number of observations required to calculate statistic.
        Setting e.g. min_periods=7 prevents early series variance instability.
    min_horizon : int, default=1
        Number of periods to shift before expanding calculation. Must be >= 1.
    prefix : str, optional
        Column name prefix. Defaults to f"{target_col}_expand".

    Returns
    -------
    pd.DataFrame with new expanding feature columns appended.
    """
    if min_horizon < 1:
        raise ValueError("min_horizon must be >= 1.")

    df = df.copy()
    prefix = prefix or f"{target_col}_expand"
    shifted = df[target_col].shift(min_horizon)

    for func in functions:
        col = f"{prefix}_{func}"
        df[col] = getattr(shifted.expanding(min_periods=min_periods), func)()

    return df


def add_ewm_features(
    df: pd.DataFrame,
    target_col: str,
    spans: Sequence[int],
    adjust: bool = False,
    min_horizon: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add exponentially-weighted moving average features.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    target_col : str
        Column to compute EWM features on.
    spans : sequence of int
        Spans for exponential weighting (e.g. [7, 30, 90]).
    adjust : bool, default=False
        If False, calculates weights exponentially decaying without dividing
        by decaying adjustment factor. Matches standard time series conventions.
    min_horizon : int, default=1
        Number of periods to shift before computing EWM. Must be >= 1.
    prefix : str, optional
        Column name prefix. Defaults to f"{target_col}_ewm".

    Returns
    -------
    pd.DataFrame with new EWM feature columns appended.
    """
    if min_horizon < 1:
        raise ValueError("min_horizon must be >= 1.")

    df = df.copy()
    prefix = prefix or f"{target_col}_ewm"
    shifted = df[target_col].shift(min_horizon)

    for span in spans:
        df[f"{prefix}_span_{span}"] = shifted.ewm(span=span, adjust=adjust).mean()

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
# Trend features
# ---------------------------------------------------------------------------

def add_linear_trend(
    df: pd.DataFrame,
    col_name: str | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add a monotonically increasing integer time index feature (0, 1, 2, ..., N-1).

    Tree-based models cannot extrapolate outside the training target range,
    flatlining at the maximum seen level. A linear time index forces tree splits
    at test time to route into the highest-trend terminal leaf, recovering the trend.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    col_name : str, optional
        Explicit column name. Defaults to 't_index' (or f'{prefix}_index' if prefix is set).
    prefix : str, optional
        Prefix for the column name if col_name is not provided.

    Returns
    -------
    pd.DataFrame with linear trend column appended.
    """
    df = df.copy()
    name = col_name or (f"{prefix}_index" if prefix else "t_index")
    df[name] = np.arange(len(df))
    return df


def add_polynomial_trend(
    df: pd.DataFrame,
    degree: int = 2,
    prefix: str = "t",
) -> pd.DataFrame:
    """
    Add polynomial trend features (e.g. t_index, t_squared, t_cubed).

    Useful for modeling accelerating or decelerating non-linear growth curves.
    Note: higher-degree polynomials extrapolate wildly outside the training range.
    In practice, degree > 2 is strongly discouraged in favor of piecewise trends.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    degree : int, default=2
        Maximum polynomial degree. Warns if degree > 2.
    prefix : str, default='t'
        Base prefix for polynomial columns.

    Returns
    -------
    pd.DataFrame with polynomial trend columns appended.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1.")

    if degree > 2:
        import warnings
        warnings.warn(
            f"Polynomial degree {degree} > 2 carries severe extrapolation risk outside the training range. "
            "Consider using add_piecewise_trend instead.",
            UserWarning,
            stacklevel=2,
        )

    df = df.copy()
    t = np.arange(len(df))
    df[f"{prefix}_index"] = t

    names_map = {2: f"{prefix}_squared", 3: f"{prefix}_cubed"}
    for d in range(2, degree + 1):
        col = names_map.get(d, f"{prefix}_pow_{d}")
        df[col] = t ** d

    return df


def add_piecewise_trend(
    df: pd.DataFrame,
    changepoints: Sequence[int],
    prefix: str = "trend",
) -> pd.DataFrame:
    """
    Add a linear time index plus knot features at each changepoint.

    Each knot feature is np.maximum(0, t - cp), which is 0 before the changepoint
    and grows linearly after, allowing the tree or linear model to learn
    different slopes for distinct trend regimes without overfitting.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically.
    changepoints : sequence of int
        Integer index positions where trend regimes change.
    prefix : str, default='trend'
        Column name prefix for changepoint knots.

    Returns
    -------
    pd.DataFrame with t_index and piecewise knot columns appended.
    """
    df = df.copy()
    t = np.arange(len(df))
    df["t_index"] = t

    for cp in changepoints:
        if cp < 0 or cp >= len(df):
            raise ValueError(f"Changepoint {cp} is out of bounds for DataFrame of length {len(df)}.")
        df[f"{prefix}_after_{cp}"] = np.maximum(0, t - cp)

    return df


def detrend_series(
    series: pd.Series,
    train_end_idx: int | None = None,
) -> tuple[pd.Series, pd.Series, object]:
    """
    Fit a linear trend on the training portion and detrend the series.

    Strictly leakage-safe: the linear regression model is fit ONLY on
    observations up to `train_end_idx`. The fitted trend is then evaluated
    over the full series length and subtracted to yield stationary residuals.

    Parameters
    ----------
    series : pd.Series
        Time series to detrend.
    train_end_idx : int, optional
        Index integer up to which data is considered training data.
        If None, fits across the full series (only suitable for final deployment).

    Returns
    -------
    tuple of (detrended_series, trend_line, linear_model)
        detrended_series : pd.Series
            Target series with linear trend subtracted (y - trend).
        trend_line : pd.Series
            Fitted and extrapolated linear trend values with identical index.
        linear_model : LinearRegression
            The fitted scikit-learn LinearRegression instance.
    """
    from sklearn.linear_model import LinearRegression

    n = len(series)
    t = np.arange(n).reshape(-1, 1)

    end_idx = train_end_idx if train_end_idx is not None else n
    if end_idx <= 0 or end_idx > n:
        raise ValueError(f"train_end_idx must be in [1, {n}], got {train_end_idx}.")

    model = LinearRegression()
    model.fit(t[:end_idx], series.values[:end_idx])

    trend_vals = model.predict(t)
    trend_line = pd.Series(trend_vals, index=series.index, name=f"{series.name or 'target'}_trend")
    detrended = pd.Series(series.values - trend_vals, index=series.index, name=f"{series.name or 'target'}_detrended")

    return detrended, trend_line, model


# ---------------------------------------------------------------------------
# Convenience: build the full leakage-safe feature matrix
# ---------------------------------------------------------------------------

def build_forecasting_feature_matrix(
    series: pd.Series,
    lags: Sequence[int] = (1, 7, 14, 28),
    rolling_windows: Sequence[int] = (7, 14, 28),
    rolling_functions: Sequence[str] = ("mean", "std"),
    expanding_functions: Sequence[str] | None = None,
    ewm_spans: Sequence[int] | None = None,
    forecast_horizon: int = 1,
    include_datetime: bool = True,
    trend: str | None = None,
) -> pd.DataFrame:
    """
    Build a complete, leakage-safe feature matrix for time series forecasting.

    Convenience wrapper combining lag, rolling, expanding, EWM, datetime, and trend features.
    All rolling, expanding, EWM, and lag operations respect the given forecast_horizon.

    Parameters
    ----------
    series : pd.Series
        Target series with DatetimeIndex.
    lags : sequence of int
        Lag periods. All lags must be >= forecast_horizon to be usable.
    rolling_windows : sequence of int
    rolling_functions : sequence of str
    expanding_functions : sequence of str, optional
        Names of expanding reductions to compute ('mean', 'std', etc.).
    ewm_spans : sequence of int, optional
        Spans for exponentially-weighted moving averages (e.g. [7, 30]).
    forecast_horizon : int
        Minimum shift applied before any rolling / lag operation.
    include_datetime : bool
    trend : {'linear', 'polynomial'} or None, optional
        Optional trend feature to include. 'linear' adds 't_index',
        'polynomial' adds 't_index' and 't_squared'.

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

    # Expanding features
    if expanding_functions:
        df = add_expanding_features(
            df, target_col, functions=expanding_functions,
            min_horizon=forecast_horizon,
        )

    # EWM features
    if ewm_spans:
        df = add_ewm_features(
            df, target_col, spans=ewm_spans,
            min_horizon=forecast_horizon,
        )

    # Datetime features
    if include_datetime:
        df = add_datetime_features(df)
        df = add_cyclical_encoding(df, "dayofweek", period=7, drop_original=False)
        df = add_cyclical_encoding(df, "month", period=12, drop_original=False)

    # Trend features
    if trend is not None:
        if trend == "linear":
            df = add_linear_trend(df)
        elif trend == "polynomial":
            df = add_polynomial_trend(df, degree=2)
        else:
            raise ValueError(f"Unsupported trend: {trend}. Choose 'linear', 'polynomial', or None.")

    return df.dropna()


# ---------------------------------------------------------------------------
# Missing Data Imputation & Missingness Features
# ---------------------------------------------------------------------------

def seasonal_fill(series: pd.Series, period: int = 7) -> pd.Series:
    """
    Fill missing values in a time series using the same-period value from previous cycles.

    Falls back to linear interpolation for remaining gaps at the start of the series.

    Parameters
    ----------
    series : pd.Series
        Time series with missing values (NaN).
    period : int, default=7
        Seasonal period length (e.g. 7 for weekly pattern on daily data).

    Returns
    -------
    pd.Series
        Series with missing values filled while preserving seasonal phase.
    """
    filled = series.copy()
    n = len(filled)
    for i in range(n):
        if pd.isna(filled.iloc[i]) and i >= period:
            lookback = period
            while lookback <= 4 * period and (i - lookback >= 0) and pd.isna(filled.iloc[i - lookback]):
                lookback += period
            if (i - lookback >= 0) and not pd.isna(filled.iloc[i - lookback]):
                filled.iloc[i] = filled.iloc[i - lookback]
    return filled.interpolate(method="linear")


def stl_imputation(series: pd.Series, period: int = 7) -> pd.Series:
    """
    Impute missing values using Seasonal and Trend decomposition using Loess (STL).

    Ideal for long/extended gaps (e.g. 21 days) where linear interpolation or simple
    forward fill destroys seasonal peaks and trend structure.

    Parameters
    ----------
    series : pd.Series
        Time series with missing values.
    period : int, default=7
        Seasonal period length for STL.

    Returns
    -------
    pd.Series
        Imputed series with trend + seasonal components reconstructed over missing gaps.
    """
    from statsmodels.tsa.seasonal import STL

    missing_mask = series.isna()
    if not missing_mask.any():
        return series.copy()

    # Step 1: Rough linear interpolation to allow STL fitting
    rough_fill = series.interpolate(method="linear").bfill().ffill()

    # Step 2: Fit STL decomposition
    stl = STL(rough_fill, period=period, robust=True).fit()

    # Step 3: Reconstruct target as trend + seasonal
    reconstructed = stl.trend + stl.seasonal

    # Step 4: Overwrite missing positions with reconstructed values
    filled = series.copy()
    filled[missing_mask] = reconstructed[missing_mask]

    return filled


def add_missingness_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Engineer explicit features capturing missingness patterns in target_col.

    Creates:
    - `{target_col}_was_missing`: Binary flag (1 if originally missing, 0 otherwise)
    - `days_since_observed`: Consecutive steps since last valid observation
    - `run_of_missing`: Current run length of consecutive missing values

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing target_col.
    target_col : str
        Column to track missingness on.

    Returns
    -------
    pd.DataFrame with missingness indicator features appended.
    """
    df = df.copy()
    is_na = df[target_col].isna()

    # 1. Binary indicator
    df[f"{target_col}_was_missing"] = is_na.astype(int)

    # 2. Days / steps since last observed
    obs_group = df[target_col].notna().cumsum()
    df["days_since_observed"] = obs_group.eq(obs_group.shift()).groupby(obs_group).cumsum()

    # 3. Current run length of missing values
    df["run_of_missing"] = is_na.astype(int).groupby(df[target_col].notna().cumsum()).cumsum()

    return df


def robust_time_series_imputation(
    series: pd.Series,
    period: int = 7,
    gap_threshold: int = 3,
) -> pd.DataFrame:
    """
    Production-ready robust time series imputation pipeline.

    Combines short-gap linear interpolation, long-gap STL reconstruction, and explicit
    missingness feature generation into a single pipeline.

    Parameters
    ----------
    series : pd.Series
        Target time series with missing values.
    period : int, default=7
        Seasonal period.
    gap_threshold : int, default=3
        Maximum gap length treated as a short gap (linear interpolation).
        Gaps > gap_threshold use STL reconstruction.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - `value`: Imputed time series values
        - `was_missing`: Binary flag indicating original missing status
        - `gap_size_at_position`: Size of the missing gap for missing entries
    """
    original_missing = series.isna().copy()

    if not original_missing.any():
        return pd.DataFrame({
            "value": series.copy(),
            "was_missing": 0,
            "gap_size_at_position": 0,
        }, index=series.index)

    # Identify continuous gap sizes
    gap_ids = original_missing.astype(int).diff().ne(0).cumsum()
    gap_sizes = original_missing.groupby(gap_ids).transform("sum")

    long_gap_mask = original_missing & (gap_sizes > gap_threshold)

    # Step 1: Linear interpolation (covers short gaps)
    filled = series.interpolate(method="linear").bfill().ffill()

    # Step 2: STL reconstruction for long gaps
    if long_gap_mask.any():
        from statsmodels.tsa.seasonal import STL
        stl = STL(filled, period=period, robust=True).fit()
        reconstructed = stl.trend + stl.seasonal
        filled[long_gap_mask] = reconstructed[long_gap_mask]

    return pd.DataFrame({
        "value": filled,
        "was_missing": original_missing.astype(int),
        "gap_size_at_position": gap_sizes.fillna(0).astype(int),
    }, index=series.index)


# ---------------------------------------------------------------------------
# Outlier & Anomaly Detection
# ---------------------------------------------------------------------------

def rolling_zscore(series: pd.Series, window: int = 28) -> pd.Series:
    """
    Compute the rolling z-score of a series. Uses shift(1) before
    rolling to prevent the current observation from contaminating
    its own baseline.

    Parameters
    ----------
    series : pd.Series
        Time series to compute z-scores on.
    window : int, default=28
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling z-scores.
    """
    rolling_mean = series.shift(1).rolling(window=window, min_periods=max(2, window // 4)).mean()
    rolling_std = series.shift(1).rolling(window=window, min_periods=max(2, window // 4)).std()
    return (series - rolling_mean) / rolling_std


def rolling_iqr_bands(
    series: pd.Series,
    window: int = 28,
    k: float = 1.5,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute rolling upper and lower anomaly bands using the IQR rule:
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    Parameters
    ----------
    series : pd.Series
        Time series to compute bands on.
    window : int, default=28
        Rolling window size.
    k : float, default=1.5
        IQR multiplier (Tukey boxplot convention).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (lower_band, upper_band)
    """
    q1 = series.shift(1).rolling(window=window, min_periods=max(4, window // 4)).quantile(0.25)
    q3 = series.shift(1).rolling(window=window, min_periods=max(4, window // 4)).quantile(0.75)
    iqr = q3 - q1
    upper = q3 + k * iqr
    lower = q1 - k * iqr
    return lower, upper


def lowess_anomalies(
    series: pd.Series,
    frac: float = 0.05,
    it: int = 0,
    threshold_sigma: float = 3.0,
) -> pd.Series:
    """
    Fit LOWESS to the series. Flag observations whose residuals
    exceed threshold_sigma standard deviations from the mean residual.

    Parameters
    ----------
    series : pd.Series
        Time series.
    frac : float, default=0.05
        Fraction of data used when estimating each local fit.
    it : int, default=0
        Number of robustifying iterations in LOWESS. Set to 0 to prevent
        extreme outliers from collapsing local weights back to self-values.
    threshold_sigma : float, default=3.0
        Number of standard deviations for residual threshold.

    Returns
    -------
    pd.Series
        Subset of residuals corresponding to detected anomalies.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    smoothed = lowess(
        series.values,
        np.arange(len(series)),
        frac=frac,
        it=it,
        return_sorted=False,
    )
    fit = pd.Series(smoothed, index=series.index)
    residuals = series - fit

    std_resid = residuals.std()
    mean_resid = residuals.mean()
    upper = mean_resid + threshold_sigma * std_resid
    lower = mean_resid - threshold_sigma * std_resid
    return residuals[(residuals > upper) | (residuals < lower)]


def stl_anomalies(
    series: pd.Series,
    period: int = 7,
    threshold_sigma: float = 3.0,
) -> pd.Series:
    """
    STL residual-based anomaly detection.
    Robust=True downweights outliers when fitting.

    Parameters
    ----------
    series : pd.Series
        Time series.
    period : int, default=7
        Seasonal period.
    threshold_sigma : float, default=3.0
        Threshold in standard deviations of the residual.

    Returns
    -------
    pd.Series
        Subset of residuals corresponding to detected anomalies.
    """
    from statsmodels.tsa.seasonal import STL

    stl = STL(series, period=period, robust=True).fit()
    residuals = stl.resid

    std_resid = residuals.std()
    mean_resid = residuals.mean()
    upper = mean_resid + threshold_sigma * std_resid
    lower = mean_resid - threshold_sigma * std_resid
    return residuals[(residuals > upper) | (residuals < lower)]


def add_anomaly_features(
    series: pd.Series,
    period: int = 7,
    threshold_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Build anomaly-related features from a time series:
    - The STL residual value itself
    - Rolling std of STL residuals (regime-shift signal)
    - Binary flag for extreme residuals
    - Days/steps since last anomaly

    Parameters
    ----------
    series : pd.Series
        Target time series.
    period : int, default=7
        Seasonal period for STL.
    threshold_sigma : float, default=3.0
        Threshold multiplier on residual std.

    Returns
    -------
    pd.DataFrame
        DataFrame with anomaly features.
    """
    from statsmodels.tsa.seasonal import STL

    stl = STL(series, period=period, robust=True).fit()
    residuals = stl.resid

    threshold = threshold_sigma * residuals.std()

    features = pd.DataFrame(index=series.index)
    features["stl_residual"] = residuals
    features["residual_volatility"] = residuals.shift(1).rolling(window=14, min_periods=2).std()
    features["is_extreme_residual"] = (residuals.abs() > threshold).astype(int)

    is_anomaly = features["is_extreme_residual"].astype(bool)
    anomaly_group = is_anomaly.cumsum()
    features["days_since_last_anomaly"] = (
        (~is_anomaly)
        .groupby(anomaly_group)
        .cumcount()
    )

    return features


