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
    add_anomaly_features,
    add_business_day_features,
    add_cyclical_encoding,
    add_datetime_features,
    add_event_features,
    add_ewm_features,
    add_expanding_features,
    add_fourier_terms,
    add_holiday_distance_features,
    add_lag_features,
    add_lag_features_horizon_aware,
    add_linear_trend,
    add_missingness_features,
    add_piecewise_trend,
    add_polynomial_trend,
    add_rolling_features,
    add_seasonal_features,
    as_categorical,
    build_forecasting_feature_matrix,
    cross_correlation,
    detrend_series,
    lowess_anomalies,
    robust_time_series_imputation,
    rolling_iqr_bands,
    rolling_zscore,
    seasonal_fill,
    stl_anomalies,
    stl_imputation,
    target_encode_expanding,
    target_encode_naive,
    target_encode_out_of_fold,
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

    def test_horizon_aware_lag_filtering(self, sample_df):
        # When horizon=7, lag_1 and lag_3 should be dropped, only lag_7 and lag_14 retained
        with pytest.warns(UserWarning, match="Dropped lags"):
            result = add_lag_features_horizon_aware(
                sample_df, "target", lags=[1, 3, 7, 14], forecast_horizon=7
            )
        assert "target_lag_7" in result.columns
        assert "target_lag_14" in result.columns
        assert "target_lag_1" not in result.columns
        assert "target_lag_3" not in result.columns

    def test_cross_correlation(self):
        # Create target and leading exogenous variable (exog leads target by 3 days)
        np.random.seed(42)
        exog = pd.Series(np.random.randn(200))
        target = exog.shift(3) + 0.1 * pd.Series(np.random.randn(200))
        
        lags, corrs = cross_correlation(target, exog, max_lag=10)
        assert len(lags) == 21
        assert 3 in lags
        peak_lag = lags[int(np.argmax(corrs))]
        assert peak_lag == 3



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

    def test_expanding_min_periods(self, sample_df):
        result = add_expanding_features(
            sample_df, "target", functions=["mean"], min_periods=7, min_horizon=1
        )
        # With min_horizon=1 and min_periods=7, indices 0..6 should be NaN
        assert result["target_expand_mean"].iloc[:7].isna().all()
        assert not np.isnan(result["target_expand_mean"].iloc[7])


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

    def test_ewm_adjust_parameter(self, sample_df):
        res_false = add_ewm_features(sample_df, "target", spans=[7], adjust=False)
        res_true = add_ewm_features(sample_df, "target", spans=[7], adjust=True)
        # Early values differ between adjust=True and adjust=False
        assert not np.isclose(
            res_false["target_ewm_span_7"].iloc[2],
            res_true["target_ewm_span_7"].iloc[2],
        )


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

    def test_fourier_values_bounded(self, sample_df):
        result = add_fourier_terms(sample_df, period=365.25, n_terms=6)
        fourier_cols = [c for c in result.columns if c.startswith("fourier_")]
        for col in fourier_cols:
            assert result[col].between(-1, 1).all(), f"{col} out of [-1, 1]"

    def test_fourier_custom_prefix(self, sample_df):
        result = add_fourier_terms(sample_df, period=365.25, n_terms=2, prefix="fy")
        assert "fy_sin_1" in result.columns
        assert "fy_cos_2" in result.columns

    def test_fourier_column_count(self, sample_df):
        result = add_fourier_terms(sample_df, period=365.25, n_terms=6)
        fourier_cols = [c for c in result.columns if c.startswith("fourier_")]
        assert len(fourier_cols) == 12  # 6 sin + 6 cos


# ---------------------------------------------------------------------------
# Seasonal features (Article 9 recipe)
# ---------------------------------------------------------------------------

class TestSeasonalityFeatures:
    """Tests for the add_seasonal_features convenience function."""

    @pytest.fixture
    def daily_df(self):
        """One year of daily data."""
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        return pd.DataFrame({"sales": np.random.default_rng(42).normal(100, 10, 365)},
                            index=dates)

    def test_default_column_count(self, daily_df):
        """Default (K=6) should produce 7 dummies + 12 Fourier = 19 new columns."""
        result = add_seasonal_features(daily_df)
        new_cols = [c for c in result.columns if c != "sales"]
        assert len(new_cols) == 19

    def test_dow_dummy_names(self, daily_df):
        result = add_seasonal_features(daily_df)
        for i in range(7):
            assert f"dow_{i}" in result.columns

    def test_fourier_column_names(self, daily_df):
        result = add_seasonal_features(daily_df)
        for k in range(1, 7):
            assert f"fy_sin_{k}" in result.columns
            assert f"fy_cos_{k}" in result.columns

    def test_no_raw_integer_dayofweek(self, daily_df):
        """The raw integer dayofweek column should be dropped, not kept."""
        result = add_seasonal_features(daily_df)
        assert "dayofweek" not in result.columns

    def test_dummies_are_binary(self, daily_df):
        result = add_seasonal_features(daily_df)
        for i in range(7):
            col = result[f"dow_{i}"]
            assert set(col.unique()) <= {0, 1}

    def test_exactly_one_dummy_active_per_row(self, daily_df):
        result = add_seasonal_features(daily_df)
        dummy_cols = [f"dow_{i}" for i in range(7)]
        assert (result[dummy_cols].sum(axis=1) == 1).all()

    def test_fourier_bounded(self, daily_df):
        result = add_seasonal_features(daily_df)
        fourier_cols = [c for c in result.columns if c.startswith("fy_")]
        for col in fourier_cols:
            assert result[col].between(-1, 1).all()

    def test_disable_dummies(self, daily_df):
        result = add_seasonal_features(daily_df, include_dow_dummies=False)
        assert not any(c.startswith("dow_") for c in result.columns)
        fourier_cols = [c for c in result.columns if c.startswith("fy_")]
        assert len(fourier_cols) == 12

    def test_disable_fourier(self, daily_df):
        result = add_seasonal_features(daily_df, yearly_k=0)
        fourier_cols = [c for c in result.columns if c.startswith("fy_")]
        assert len(fourier_cols) == 0
        assert any(c.startswith("dow_") for c in result.columns)

    def test_custom_fourier_k(self, daily_df):
        result = add_seasonal_features(daily_df, yearly_k=3)
        fourier_cols = [c for c in result.columns if c.startswith("fy_")]
        assert len(fourier_cols) == 6  # 3 sin + 3 cos

    def test_custom_fourier_prefix(self, daily_df):
        result = add_seasonal_features(daily_df, fourier_prefix="yearly")
        assert any(c.startswith("yearly_") for c in result.columns)
        assert not any(c.startswith("fy_") for c in result.columns)

    def test_original_columns_preserved(self, daily_df):
        result = add_seasonal_features(daily_df)
        assert "sales" in result.columns
        assert len(result) == len(daily_df)

    def test_leakage_free(self, daily_df):
        """Seasonal features are deterministic functions of the timestamp,
        so they should be identical regardless of the target values."""
        result1 = add_seasonal_features(daily_df)
        df2 = daily_df.copy()
        df2["sales"] = 0  # zero out target
        result2 = add_seasonal_features(df2)
        seasonal_cols = [c for c in result1.columns if c != "sales"]
        for col in seasonal_cols:
            assert (result1[col] == result2[col]).all(), (
                f"{col} changed when target was zeroed — leakage!"
            )


# ---------------------------------------------------------------------------
# Trend features
# ---------------------------------------------------------------------------

class TestTrendFeatures:
    def test_linear_trend_correctness(self, sample_df):
        result = add_linear_trend(sample_df)
        assert "t_index" in result.columns
        assert (result["t_index"].values == np.arange(len(sample_df))).all()

    def test_linear_trend_custom_name(self, sample_df):
        res1 = add_linear_trend(sample_df, col_name="time_step")
        assert "time_step" in res1.columns

        res2 = add_linear_trend(sample_df, prefix="custom")
        assert "custom_index" in res2.columns

    def test_polynomial_trend_correctness(self, sample_df):
        result = add_polynomial_trend(sample_df, degree=2)
        assert "t_index" in result.columns
        assert "t_squared" in result.columns
        assert (result["t_squared"].values == np.arange(len(sample_df)) ** 2).all()

    def test_polynomial_trend_warnings_and_errors(self, sample_df):
        with pytest.raises(ValueError, match="degree must be >= 1"):
            add_polynomial_trend(sample_df, degree=0)

        with pytest.warns(UserWarning, match="Polynomial degree 3 > 2"):
            res3 = add_polynomial_trend(sample_df, degree=3)
            assert "t_cubed" in res3.columns

    def test_piecewise_trend_correctness(self, sample_df):
        changepoints = [20, 50]
        result = add_piecewise_trend(sample_df, changepoints=changepoints)
        assert "t_index" in result.columns
        assert "trend_after_20" in result.columns
        assert "trend_after_50" in result.columns

        # Before changepoint, knot is strictly 0
        assert (result.iloc[:21]["trend_after_20"] == 0).all()
        # After changepoint, knot equals t - cp
        assert result.iloc[25]["trend_after_20"] == 5
        assert result.iloc[60]["trend_after_50"] == 10

    def test_piecewise_trend_out_of_bounds(self, sample_df):
        with pytest.raises(ValueError, match="out of bounds"):
            add_piecewise_trend(sample_df, changepoints=[200])

    def test_detrend_series_correctness_and_leakage(self, sample_series):
        # Monotonic linear series with slope=2, intercept=10
        dates = sample_series.index
        t = np.arange(len(dates))
        y = pd.Series(10.0 + 2.0 * t, index=dates, name="sales")

        # Fit only on first 70 observations
        detrended, trend_line, model = detrend_series(y, train_end_idx=70)

        # 1. Model slope should be 2.0
        assert pytest.approx(model.coef_[0]) == 2.0
        assert pytest.approx(model.intercept_) == 10.0

        # 2. Detrended series on training split has near-zero residuals
        assert pytest.approx(detrended.iloc[:70].abs().max(), abs=1e-6) == 0.0

        # 3. Detrended series on extrapolated test split also has near-zero residuals
        # because ground truth was pure linear
        assert pytest.approx(detrended.iloc[70:].abs().max(), abs=1e-6) == 0.0

        # 4. Length and indices preserved
        assert len(detrended) == len(y)
        assert (detrended.index == y.index).all()
        assert (trend_line.index == y.index).all()

    def test_detrend_series_invalid_train_end(self, sample_series):
        with pytest.raises(ValueError, match="train_end_idx"):
            detrend_series(sample_series, train_end_idx=0)
        with pytest.raises(ValueError, match="train_end_idx"):
            detrend_series(sample_series, train_end_idx=len(sample_series) + 10)


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

    def test_expanding_and_ewm_in_feature_matrix(self, sample_series):
        df = build_forecasting_feature_matrix(
            sample_series,
            lags=[1, 7],
            rolling_windows=[7],
            expanding_functions=["mean", "std"],
            ewm_spans=[7, 30],
        )
        assert "target_expand_mean" in df.columns
        assert "target_expand_std" in df.columns
        assert "target_ewm_span_7" in df.columns
        assert "target_ewm_span_30" in df.columns

    def test_trend_in_feature_matrix(self, sample_series):
        # Linear trend
        df_lin = build_forecasting_feature_matrix(
            sample_series, lags=[1, 7], rolling_windows=[7], trend="linear"
        )
        assert "t_index" in df_lin.columns

        # Polynomial trend
        df_poly = build_forecasting_feature_matrix(
            sample_series, lags=[1, 7], rolling_windows=[7], trend="polynomial"
        )
        assert "t_index" in df_poly.columns
        assert "t_squared" in df_poly.columns

        # Invalid trend
        with pytest.raises(ValueError, match="Unsupported trend"):
            build_forecasting_feature_matrix(
                sample_series, lags=[1, 7], rolling_windows=[7], trend="unsupported"
            )


# ---------------------------------------------------------------------------
# Missing Data Tests
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_seasonal_fill(self, sample_series):
        s = sample_series.copy()
        s.iloc[10] = np.nan  # missing at i=10 (should be filled from i=3)
        filled = seasonal_fill(s, period=7)
        assert not filled.isna().any()
        assert filled.iloc[10] == s.iloc[3]

    def test_stl_imputation(self, sample_series):
        s = sample_series.copy()
        s.iloc[20:25] = np.nan
        filled = stl_imputation(s, period=7)
        assert not filled.isna().any()

    def test_add_missingness_features(self, sample_df):
        df = sample_df.copy()
        df.iloc[5:8, df.columns.get_loc("target")] = np.nan
        res = add_missingness_features(df, "target")
        assert "target_was_missing" in res.columns
        assert "days_since_observed" in res.columns
        assert "run_of_missing" in res.columns
        assert res.iloc[5]["target_was_missing"] == 1
        assert res.iloc[0]["target_was_missing"] == 0

    def test_robust_time_series_imputation(self, sample_series):
        s = sample_series.copy()
        s.iloc[10:12] = np.nan  # short gap (2)
        s.iloc[30:37] = np.nan  # long gap (7)
        res = robust_time_series_imputation(s, period=7, gap_threshold=3)
        assert not res["value"].isna().any()
        assert "was_missing" in res.columns
        assert "gap_size_at_position" in res.columns


# ---------------------------------------------------------------------------
# Outlier / Anomaly Detection Tests
# ---------------------------------------------------------------------------

class TestOutlierDetection:
    def test_rolling_zscore_detects_spike(self, sample_series):
        s = sample_series.copy()
        s.iloc[50] = 500.0  # massive spike
        z = rolling_zscore(s, window=28)
        assert abs(z.iloc[50]) > 3.0

    def test_rolling_iqr_bands(self, sample_series):
        s = sample_series.copy()
        s.iloc[50] = 500.0
        lower, upper = rolling_iqr_bands(s, window=28, k=1.5)
        assert s.iloc[50] > upper.iloc[50]

    def test_lowess_anomalies(self, sample_series):
        s = sample_series.copy()
        s.iloc[50] = 500.0
        anomalies = lowess_anomalies(s, frac=0.1, threshold_sigma=3.0)
        assert len(anomalies) >= 1
        assert s.index[50] in anomalies.index

    def test_stl_anomalies(self, sample_series):
        # Create series with weekly seasonality and injected spike
        dates = pd.date_range("2024-01-01", periods=140, freq="D")
        t = np.arange(140)
        seasonal = 10 * np.sin(2 * np.pi * t / 7)
        values = 50 + seasonal + np.random.normal(0, 1, 140)
        values[70] = 150.0  # anomaly
        s = pd.Series(values, index=dates)
        anomalies = stl_anomalies(s, period=7, threshold_sigma=3.0)
        assert len(anomalies) >= 1
        assert dates[70] in anomalies.index

    def test_add_anomaly_features(self, sample_series):
        dates = pd.date_range("2024-01-01", periods=140, freq="D")
        s = pd.Series(np.arange(140, dtype=float), index=dates)
        feats = add_anomaly_features(s, period=7, threshold_sigma=3.0)
        assert "stl_residual" in feats.columns
        assert "residual_volatility" in feats.columns
        assert "is_extreme_residual" in feats.columns
        assert "days_since_last_anomaly" in feats.columns


# ---------------------------------------------------------------------------
# Event & calendar features (Article 10)
# ---------------------------------------------------------------------------

class TestEventFeatures:
    @pytest.fixture
    def calendar_df(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        return pd.DataFrame({"date": dates, "sales": np.arange(60, dtype=float)})

    @pytest.fixture
    def holidays(self) -> pd.DataFrame:
        # Jan 1 and Jan 31 of the same range as calendar_df
        return pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-31"])})

    def test_is_holiday_flag(self, calendar_df, holidays):
        res = add_holiday_distance_features(calendar_df, holidays, date_col="date")
        assert res.loc[res["date"] == "2024-01-01", "is_holiday"].item() == 1
        assert res.loc[res["date"] == "2024-01-15", "is_holiday"].item() == 0

    def test_days_to_and_since_holiday(self, calendar_df, holidays):
        res = add_holiday_distance_features(calendar_df, holidays, date_col="date")
        row = res.loc[res["date"] == "2024-01-10"].iloc[0]
        # 2024-01-10 is 21 days before Jan 31 and 9 days after Jan 1
        assert row["days_to_holiday"] == 21
        assert row["days_since_holiday"] == 9

    def test_holiday_distance_is_clipped(self, calendar_df, holidays):
        res = add_holiday_distance_features(calendar_df, holidays, date_col="date", max_distance=5)
        assert (res["days_to_holiday"] <= 5).all()
        assert (res["days_since_holiday"] <= 5).all()

    def test_holiday_features_use_only_known_calendar(self, calendar_df, holidays):
        """The calendar itself carries no information from `sales` — the
        columns are a function of date and the holiday list alone, so they
        are identical regardless of what the target does."""
        shuffled = calendar_df.copy()
        shuffled["sales"] = shuffled["sales"].sample(frac=1, random_state=0).to_numpy()
        res_a = add_holiday_distance_features(calendar_df, holidays, date_col="date")
        res_b = add_holiday_distance_features(shuffled, holidays, date_col="date")
        cols = ["is_holiday", "days_to_holiday", "days_since_holiday"]
        pd.testing.assert_frame_equal(res_a[cols], res_b[cols])

    def test_business_day_month_end(self, calendar_df):
        res = add_business_day_features(calendar_df, date_col="date")
        # Jan 2024: the 31st is a Wednesday, a business day itself
        assert res.loc[res["date"] == "2024-01-31", "is_month_end_bday"].item() == 1
        assert res.loc[res["date"] == "2024-01-31", "bdays_to_month_end"].item() == 0
        assert res.loc[res["date"] == "2024-01-01", "bdays_to_month_end"].item() > 0

    def test_add_event_features_combines_both(self, calendar_df, holidays):
        res = add_event_features(calendar_df, holidays, date_col="date")
        for col in ["is_holiday", "days_to_holiday", "days_since_holiday",
                    "is_month_end_bday", "bdays_to_month_end"]:
            assert col in res.columns

    def test_holidays_requires_at_least_one_date(self, calendar_df):
        with pytest.raises(ValueError):
            add_holiday_distance_features(calendar_df, pd.DataFrame({"date": []}), date_col="date")


# ---------------------------------------------------------------------------
# Categorical encoding for panel data (Article 10)
# ---------------------------------------------------------------------------

class TestCategoricalEncoding:
    @pytest.fixture
    def panel_df(self) -> pd.DataFrame:
        """Two stores, 10 days each, sorted by [store, date] — a deterministic
        target so exact encoding values can be checked by hand."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        rows = []
        for store, base in [("A", 10.0), ("B", 100.0)]:
            for i, d in enumerate(dates):
                rows.append({"store": store, "date": d, "y": base + i})
        return pd.DataFrame(rows).sort_values(["store", "date"]).reset_index(drop=True)

    def test_as_categorical_dtype(self, panel_df):
        res = as_categorical(panel_df, ["store"])
        assert str(res["store"].dtype) == "category"

    def test_naive_encoding_includes_own_row(self, panel_df):
        res = target_encode_naive(panel_df, "store", "y")
        expected_a = panel_df.loc[panel_df["store"] == "A", "y"].mean()
        assert np.isclose(res.loc[res["store"] == "A", "store_te_naive"].iloc[0], expected_a)

    def test_expanding_encoding_excludes_own_row(self, panel_df):
        res = target_encode_expanding(panel_df, "store", "y")
        store_a = res[res["store"] == "A"].reset_index(drop=True)
        # Row 0 of store A has no prior observation -> falls back to global prior
        global_prior = panel_df["y"].mean()
        assert np.isclose(store_a.loc[0, "store_te_expanding"], global_prior)
        # Row 2 (y=12) should equal the mean of rows 0-1 (y=10, 11) = 10.5
        assert np.isclose(store_a.loc[2, "store_te_expanding"], 10.5)

    def test_expanding_encoding_never_leaks_current_target(self, panel_df):
        """For every row, the expanding-encoded value must be computable from
        strictly earlier rows of the same category only."""
        res = target_encode_expanding(panel_df, "store", "y")
        for store, grp in res.groupby("store"):
            grp = grp.reset_index(drop=True)
            for i in range(1, len(grp)):
                expected = grp.loc[:i - 1, "y"].mean()
                assert np.isclose(grp.loc[i, "store_te_expanding"], expected)

    def test_expanding_encoding_cold_start_unseen_category(self, panel_df):
        """A category never seen before in the frame falls back to the
        global prior (computed over the frame given) rather than raising or
        returning NaN."""
        new_row = pd.DataFrame({"store": ["C"], "date": [pd.Timestamp("2024-01-11")], "y": [999.0]})
        combined = pd.concat([panel_df, new_row], ignore_index=True)
        res = target_encode_expanding(combined, "store", "y")
        assert np.isclose(res.iloc[-1]["store_te_expanding"], combined["y"].mean())

    def test_smoothing_pulls_sparse_levels_toward_global_mean(self, panel_df):
        unsmoothed = target_encode_expanding(panel_df, "store", "y", smoothing=0.0)
        smoothed = target_encode_expanding(panel_df, "store", "y", smoothing=50.0)
        global_prior = panel_df["y"].mean()
        # With heavy smoothing, later rows should sit closer to the global
        # prior than the unsmoothed expanding mean does.
        row = 5
        store_a_unsmoothed = unsmoothed[unsmoothed["store"] == "A"].reset_index(drop=True)
        store_a_smoothed = smoothed[smoothed["store"] == "A"].reset_index(drop=True)
        d_unsmoothed = abs(store_a_unsmoothed.loc[row, "store_te_expanding"] - global_prior)
        d_smoothed = abs(store_a_smoothed.loc[row, "store_te_expanding"] - global_prior)
        assert d_smoothed < d_unsmoothed

    def test_out_of_fold_encoding_fits_on_train_rows_only(self, panel_df):
        # panel_df is sorted [store, date]: rows 0-9 = store A (10 days),
        # rows 10-19 = store B (10 days). Split each store's own first 5
        # days into train and last 5 into val, so both folds contain both
        # categories.
        train_idx = np.concatenate([np.arange(0, 5), np.arange(10, 15)])
        val_idx = np.concatenate([np.arange(5, 10), np.arange(15, 20)])
        folds = [(train_idx, val_idx)]
        res = target_encode_out_of_fold(panel_df, "store", "y", folds=folds)

        train = panel_df.iloc[train_idx]
        expected_map = train.groupby("store")["y"].mean()

        val = res.iloc[val_idx]
        for _, row in val.iterrows():
            assert np.isclose(row["store_te_oof"], expected_map[row["store"]])
        # Rows outside any fold's val_idx are left as NaN
        assert res.iloc[train_idx]["store_te_oof"].isna().all()

    def test_out_of_fold_encoding_unseen_category_falls_back(self, panel_df):
        n = len(panel_df)
        folds = [(np.arange(0, n - 1), np.array([n - 1]))]
        df = panel_df.copy()
        df.loc[df.index[-1], "store"] = "UNSEEN"
        res = target_encode_out_of_fold(df, "store", "y", folds=folds)
        train_prior = df.iloc[: n - 1]["y"].mean()
        assert np.isclose(res.iloc[-1]["store_te_oof"], train_prior)


