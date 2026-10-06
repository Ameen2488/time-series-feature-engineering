"""
Data loading utilities.

Provides:
- Synthetic time series generators for reproducible demonstrations
- Loaders for public datasets referenced in the notebooks
"""

from __future__ import annotations

from typing import Sequence

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


def generate_holiday_calendar(years: Sequence[int]) -> pd.DataFrame:
    """
    Build a small synthetic retail holiday calendar.

    Includes fixed-date holidays (New Year's Day, Independence Day, Christmas)
    and floating holidays computed from a rule (Thanksgiving = 4th Thursday of
    November, "Black Friday" = 4th Friday of November). The floating dates are
    *computed*, not hand-typed, so they land correctly in every year and shift
    day-of-year the way real floating holidays do — which is exactly why a
    day-of-year Fourier term cannot represent them.

    Parameters
    ----------
    years : sequence of int
        Calendar years to generate holidays for.

    Returns
    -------
    pd.DataFrame with columns ['date', 'name'], sorted by date.
    """
    rows = []
    for year in years:
        rows.append((pd.Timestamp(year, 1, 1), "new_years_day"))
        rows.append((pd.Timestamp(year, 7, 4), "independence_day"))
        rows.append((pd.Timestamp(year, 12, 25), "christmas"))

        # 4th Thursday of November
        nov1 = pd.Timestamp(year, 11, 1)
        first_thu = nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7)
        thanksgiving = first_thu + pd.Timedelta(weeks=3)
        rows.append((thanksgiving, "thanksgiving"))
        rows.append((thanksgiving + pd.Timedelta(days=1), "black_friday"))

    cal = pd.DataFrame(rows, columns=["date", "name"]).sort_values("date")
    return cal.reset_index(drop=True)


def generate_store_panel(
    n_stores: int = 200,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    black_friday_boost: float = 0.9,
    other_holiday_boost: float = 0.3,
    payday_boost: float = 0.08,
    noise_std_frac: float = 0.08,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic daily sales panel across many stores, with event
    effects baked in by construction: a ramp-up / spike / dip around each
    holiday (largest around "Black Friday"), a smaller bump around the other
    calendar holidays, and a mild bump near the business-day month-end
    (a stand-in for payday-driven demand).

    This is the generator behind Article 10's capstone example: 200 stores,
    3 full years (2022-2024 inclusive = 1,096 days), a synthetic "Black
    Friday" built from a rule (4th Friday of November) rather than hardcoded
    dates, so it lands on a different day-of-year every year — the property
    that breaks day-of-year Fourier terms and motivates event features.

    Parameters
    ----------
    n_stores : int, default=200
    start_date, end_date : str
        Inclusive date range.
    black_friday_boost : float, default=0.9
        Peak fractional demand multiplier on Black Friday itself
        (0.9 = +90% vs. the store's baseline that day).
    other_holiday_boost : float, default=0.3
        Peak fractional demand multiplier for the other calendar holidays.
    payday_boost : float, default=0.08
        Fractional demand multiplier at the business-day month-end.
    noise_std_frac : float, default=0.08
        Multiplicative noise standard deviation, as a fraction of each
        store's baseline level.
    seed : int

    Returns
    -------
    (panel, holidays) : tuple of pd.DataFrame
        panel : columns ['date', 'store_id', 'sales'], long format,
            sorted by ['store_id', 'date'].
        holidays : columns ['date', 'name'], as returned by
            `generate_holiday_calendar`.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    years = sorted(dates.year.unique().tolist())
    holidays = generate_holiday_calendar(years)
    bf_dates = holidays.loc[holidays["name"] == "black_friday", "date"].to_numpy()
    other_dates = holidays.loc[holidays["name"] != "black_friday", "date"].to_numpy()

    def event_kernel(target_dates: np.ndarray, peak: float, ramp: int = 10, decay: int = 4) -> np.ndarray:
        """Asymmetric bump: a slow ramp-up before the date, a spike on the
        date, and a faster dip immediately after — not a symmetric kernel,
        because real pre-event demand build-up and post-event drop-off
        are not mirror images of each other."""
        dist = (dates.values[:, None] - target_dates[None, :]) / np.timedelta64(1, "D")
        before = np.exp(-0.5 * (dist / ramp) ** 2) * (dist <= 0)
        after = np.exp(-0.5 * (dist / decay) ** 2) * (dist > 0)
        return peak * (before + after).max(axis=1)

    black_friday_mult = event_kernel(bf_dates, black_friday_boost, ramp=10, decay=4)
    other_holiday_mult = event_kernel(other_dates, other_holiday_boost, ramp=4, decay=2)

    bmonth_end = (dates + pd.offsets.BMonthEnd(0))
    bdays_to_end = np.array([np.busday_count(d.date(), e.date()) for d, e in zip(dates, bmonth_end)])
    payday_mult = payday_boost * np.exp(-0.5 * (bdays_to_end / 2.0) ** 2)

    weekly = 1.0 + 0.18 * np.sin(2 * np.pi * (dow - 4) / 7)
    yearly = 1.0 + 0.10 * np.sin(2 * np.pi * (doy - 60) / 365.25)
    event_mult = 1.0 + black_friday_mult + other_holiday_mult + payday_mult

    frames = []
    for s in range(n_stores):
        store_id = f"S{s:04d}"
        base = rng.uniform(40.0, 260.0)
        trend_slope = rng.uniform(-0.01, 0.03)
        store_sensitivity = rng.uniform(0.6, 1.4)  # how hard this store reacts to events

        level = base + trend_slope * np.arange(n_days)
        mult = weekly * yearly * (1.0 + store_sensitivity * (event_mult - 1.0))
        noise = rng.normal(1.0, noise_std_frac, n_days)

        sales = np.clip(level * mult * noise, a_min=0.0, a_max=None)
        frames.append(pd.DataFrame({"date": dates, "store_id": store_id, "sales": sales}))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["store_id", "date"]).reset_index(drop=True)
    return panel, holidays


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
