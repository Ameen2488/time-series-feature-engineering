"""
Data loading utilities.

Provides:
- Synthetic time series generators for reproducible demonstrations
- Loaders for public datasets referenced in the notebooks
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_daily_sales(
    n_days: int = 730,
    start_date: str = "2023-01-01",
    trend_slope: float = 0.05,
    weekly_amplitude: float = 15.0,
    yearly_amplitude: float = 30.0,
    noise_std: float = 5.0,
    base_level: float = 100.0,
    seed: int = 42,
) -> pd.Series:
    """
    Generate a realistic daily sales series with trend, weekly + yearly
    seasonality, and Gaussian noise.

    Parameters
    ----------
    n_days : int
        Number of days to generate.
    start_date : str
        ISO-format start date.
    trend_slope : float
        Linear trend per day.
    weekly_amplitude : float
        Amplitude of weekly seasonality (day-of-week effect).
    yearly_amplitude : float
        Amplitude of yearly seasonality.
    noise_std : float
        Standard deviation of Gaussian noise.
    base_level : float
        Series mean level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.Series with DatetimeIndex, name='sales'.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    trend = trend_slope * t
    weekly = weekly_amplitude * np.sin(2 * np.pi * t / 7)
    yearly = yearly_amplitude * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, noise_std, n_days)

    values = base_level + trend + weekly + yearly + noise
    return pd.Series(values, index=dates, name="sales")


def generate_intermittent_demand(
    n_days: int = 730,
    start_date: str = "2023-01-01",
    demand_probability: float = 0.15,
    demand_mean: float = 50.0,
    demand_std: float = 15.0,
    seed: int = 42,
) -> pd.Series:
    """
    Generate an intermittent demand series (Croston-style).

    Long stretches of zero demand punctuated by non-zero orders — typical
    of MRO parts, specialty chemicals, or slow-moving SKUs.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    has_demand = rng.random(n_days) < demand_probability
    demand_values = rng.normal(demand_mean, demand_std, n_days).clip(min=0)

    series = np.where(has_demand, demand_values, 0.0)
    return pd.Series(series, index=dates, name="demand")


def generate_multi_sku_panel(
    n_skus: int = 5,
    n_days: int = 730,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a long-format panel of daily sales for multiple SKUs, each with
    slightly different trend and seasonality parameters.

    Returns
    -------
    pd.DataFrame with columns ['date', 'sku', 'sales'].
    """
    rng = np.random.default_rng(seed)
    frames = []

    for i in range(n_skus):
        series = generate_daily_sales(
            n_days=n_days,
            start_date=start_date,
            trend_slope=rng.uniform(-0.05, 0.15),
            weekly_amplitude=rng.uniform(5, 25),
            yearly_amplitude=rng.uniform(10, 40),
            noise_std=rng.uniform(3, 8),
            base_level=rng.uniform(50, 200),
            seed=seed + i,
        )
        df = series.reset_index()
        df.columns = ["date", "sales"]
        df["sku"] = f"SKU_{i:03d}"
        frames.append(df)

    return pd.concat(frames, ignore_index=True)[["date", "sku", "sales"]]


def time_train_test_split(
    df: pd.DataFrame | pd.Series,
    test_size: float | int = 0.2,
) -> tuple:
    """
    Split a DatetimeIndex-ordered DataFrame or Series by time, without shuffling.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Must be sorted chronologically (or have a DatetimeIndex).
    test_size : float or int
        If float in (0, 1), fraction of rows for the test set.
        If int, number of rows for the test set.

    Returns
    -------
    (train, test) tuple with the same type as input.
    """
    if isinstance(test_size, float):
        if not (0 < test_size < 1):
            raise ValueError("test_size as float must be in (0, 1).")
        split_idx = int(len(df) * (1 - test_size))
    else:
        split_idx = len(df) - test_size

    return df.iloc[:split_idx], df.iloc[split_idx:]
