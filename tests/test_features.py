"""
Tests for feature engineering utilities.

The most important tests are the leakage tests — they verify that a feature
computed at time t does not use the target value at time t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_cyclical_encoding,
    add_datetime_features,
    add_ewm_features,
    add_expanding_features,
    add_fourier_terms,
    add_lag_features,
    add_rolling_features,
    build_forecasting_feature_matrix,
)


@pytest.fixture
def sample_series() -> pd.Series:
    """A simple deterministic series for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    values = np.arange(100, dtype=float)  # 0, 1, 2, ..., 99
    return pd.Series(values, index=dates, name="target")


@pytest.fixture
def sample_df(sample_series) -> pd.DataFrame:
    return sample_series.to_frame()


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

class TestLagFeatures:
    def test_lag_value_correctness(self, sample_df):
        result = add_lag_features(sample_df, "target", lags=[1, 7])
        # Row at index 10 should have target=10, lag_1=9, lag_7=3
        assert result.iloc[10]["target"] == 10
        assert result.iloc[10]["target_lag_1"] == 9
        assert result.iloc[10]["target_lag_7"] == 3

    def test_lag_leakage_impossible(self, sample_df):
        """A lag feature at time t must never equal target at time t."""
        result = add_lag_features(sample_df, "target", lags=[1, 3, 7])
        for lag_col in ["target_lag_1", "target_lag_3", "target_lag_7"]:
            valid = result.dropna(subset=[lag_col])
            assert not (valid["target"] == valid[lag_col]).any(), (
                f"Leakage detected in {lag_col}"
            )

    def test_negative_lag_rejected(self, sample_df):
        with pytest.raises(ValueError):
            add_lag_features(sample_df, "target", lags=[-1])


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

class TestRollingFeatures:
    def test_rolling_mean_correctness(self, sample_df):
        result = add_rolling_features(
            sample_df, "target", windows=[3], functions=["mean"], min_horizon=1
        )
        # At index 10: rolling mean of target[7:10] = mean(7,8,9) = 8
        expected = np.mean([7, 8, 9])
        assert result.iloc[10]["target_roll_mean_3"] == pytest.approx(expected)

    def test_rolling_leakage_impossible(self, sample_df):
        """Rolling feature at time t must never include target(t)."""
        result = add_rolling_features(
            sample_df, "target", windows=[1], functions=["mean"], min_horizon=1
        )
        # With window=1 and min_horizon=1, the rolling mean at t == target(t-1)
        valid = result.dropna(subset=["target_roll_mean_1"])
        assert (valid["target_roll_mean_1"] < valid["target"]).all()

    def test_min_horizon_respected(self, sample_df):
        """min_horizon=5 means the rolling window ends at t-5."""
        result = add_rolling_features(
            sample_df, "target", windows=[3], functions=["mean"], min_horizon=5
        )
        # At index 20: rolling mean uses target[13:16] = mean(13,14,15) = 14
        expected = np.mean([13, 14, 15])
        assert result.iloc[20]["target_roll_mean_3"] == pytest.approx(expected)

    def test_zero_min_horizon_rejected(self, sample_df):
        with pytest.raises(ValueError):
            add_rolling_features(
                sample_df, "target", windows=[7], min_horizon=0
            )


# ---------------------------------------------------------------------------
# Expanding features
# ---------------------------------------------------------------------------

class TestExpandingFeatures:
    def test_expanding_leakage_impossible(self, sample_df):
        result = add_expanding_features(
            sample_df, "target", functions=["mean"], min_horizon=1
        )
        # Expanding mean at t uses target[0:t], so must be strictly < target[t]
        # for our monotonic increasing series
        valid = result.dropna(subset=["target_expand_mean"]).iloc[1:]
        assert (valid["target_expand_mean"] < valid["target"]).all()


# ---------------------------------------------------------------------------
# EWM features
# ---------------------------------------------------------------------------

class TestEWMFeatures:
    def test_ewm_leakage_impossible(self, sample_df):
        result = add_ewm_features(
            sample_df, "target", spans=[7], min_horizon=1
        )
        valid = result.dropna(subset=["target_ewm_span_7"]).iloc[7:]
        # EWM at t uses target[0:t-1], strictly < target[t] for increasing series
        assert (valid["target_ewm_span_7"] < valid["target"]).all()


# ---------------------------------------------------------------------------
# Datetime & cyclical
# ---------------------------------------------------------------------------

class TestDatetimeFeatures:
    def test_dayofweek_extracted(self, sample_df):
        result = add_datetime_features(sample_df, features=["dayofweek"])
        assert "dayofweek" in result.columns
        # 2024-01-01 is a Monday (dayofweek=0)
        assert result.iloc[0]["dayofweek"] == 0

    def test_cyclical_encoding_range(self, sample_df):
        df = add_datetime_features(sample_df, features=["dayofweek"])
        df = add_cyclical_encoding(df, "dayofweek", period=7)
        assert df["dayofweek_sin"].between(-1, 1).all()
        assert df["dayofweek_cos"].between(-1, 1).all()


class TestFourierTerms:
    def test_fourier_terms_added(self, sample_df):
        result = add_fourier_terms(sample_df, period=365.25, n_terms=3)
        for k in range(1, 4):
            assert f"fourier_sin_{k}" in result.columns
            assert f"fourier_cos_{k}" in result.columns


# ---------------------------------------------------------------------------
# End-to-end feature matrix
# ---------------------------------------------------------------------------

class TestFeatureMatrix:
    def test_no_nulls_in_output(self, sample_series):
        df = build_forecasting_feature_matrix(
            sample_series, lags=[1, 7], rolling_windows=[7]
        )
        assert not df.isnull().any().any()

    def test_target_column_present(self, sample_series):
        df = build_forecasting_feature_matrix(sample_series)
        assert "target" in df.columns

    def test_forecast_horizon_drops_short_lags(self, sample_series):
        df = build_forecasting_feature_matrix(
            sample_series, lags=[1, 7, 14], forecast_horizon=7
        )
        # lag_1 should be dropped (< horizon of 7)
        assert "target_lag_1" not in df.columns
        assert "target_lag_7" in df.columns
        assert "target_lag_14" in df.columns
