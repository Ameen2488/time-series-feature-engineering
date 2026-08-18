# Article ↔ Code Map

Detailed cross-reference between each article in the series, the notebook
that accompanies it, and the specific `src/` utilities it uses.

---

## Article 1 — Tabularizing Time Series

- **Notebook:** [`notebooks/01_tabularizing_time_series.ipynb`](../notebooks/01_tabularizing_time_series.ipynb)
- **Read:** [Medium](https://medium.com/@asidd24/tabularizing-time-series-the-foundation-of-ml-based-forecasting-2070d22651ef) · [Newsletter](https://www.linkedin.com/newsletters/time-series-engineered-7485187061080137728/)
- **Utilities used:**
  - `src.data.generate_daily_sales`
  - `src.data.time_train_test_split`
  - `src.features.add_lag_features`
  - `src.features.add_rolling_features`
  - `src.features.add_datetime_features`
  - `src.features.add_cyclical_encoding`
  - `src.features.build_forecasting_feature_matrix`
  - `src.metrics.mae`, `rmse`, `wmape`, `vandeput_score`

## Article 2 — Why Time Series Breaks Your ML Pipeline

- **Notebook:** [`notebooks/02_ml_pipeline_leakage.ipynb`](../notebooks/02_ml_pipeline_leakage.ipynb)
- **Read:** [Medium](https://medium.com/@asidd24/why-time-series-breaks-your-ml-pipeline-and-how-to-fix-it-3177886faa54) · [Newsletter](https://www.linkedin.com/newsletters/time-series-engineered-7485187061080137728/)
- **Utilities used:** `src.validation.make_walk_forward_splitter`, `src.validation.describe_splits`

## Article 3 — Decomposing Time Series
- **Notebook:** [`notebooks/03_decomposition.ipynb`](../notebooks/03_decomposition.ipynb)

## Article 4 — Missing Data
- **Notebook:** [`notebooks/04_missing_data.ipynb`](../notebooks/04_missing_data.ipynb)
- **Read:** [Medium](https://medium.com/@asidd24/missing-data-in-time-series-beyond-simple-imputation-edae6e2543c7) · [Newsletter](https://www.linkedin.com/newsletters/time-series-engineered-7485187061080137728/)
- **Utilities used:**
  - `src.features.seasonal_fill`
  - `src.features.stl_imputation`
  - `src.features.add_missingness_features`
  - `src.features.robust_time_series_imputation`

## Article 5 — Outlier Detection

- **Notebook:** [`notebooks/05_outlier_detection.ipynb`](../notebooks/05_outlier_detection.ipynb)
- **Read:** [Medium](https://medium.com/@asidd24/outlier-detection-in-time-series-rolling-stats-lowess-stl-bc2af56a6276) · [Newsletter](https://www.linkedin.com/newsletters/time-series-engineered-7485187061080137728/)
- **Utilities used:**
  - `src.features.rolling_zscore`
  - `src.features.rolling_iqr_bands`
  - `src.features.lowess_anomalies`
  - `src.features.stl_anomalies`
  - `src.features.add_anomaly_features`

## Article 6 — Lag Features
- **Notebook:** [`notebooks/06_lag_features.ipynb`](../notebooks/06_lag_features.ipynb)

## Article 7 — Window Features
- **Notebook:** [`notebooks/07_window_features.ipynb`](../notebooks/07_window_features.ipynb)

## Article 8 — Trend Features
- **Notebook:** [`notebooks/08_trend_features.ipynb`](../notebooks/08_trend_features.ipynb)

## Article 9 — Seasonality Features
- **Notebook:** [`notebooks/09_seasonality.ipynb`](../notebooks/09_seasonality.ipynb)

## Article 10 — Datetime & Categorical Features
- **Notebook:** [`notebooks/10_datetime_categorical.ipynb`](../notebooks/10_datetime_categorical.ipynb)
