# Time Series Feature Engineering

**Production-grade feature engineering for time series forecasting and anomaly detection — with runnable code, real datasets, and no leakage.**

This is the companion code repository for a 10-part article series covering the feature engineering techniques that separate a demo notebook from a forecasting model that survives production.

Each article in the series is paired with a self-contained Jupyter notebook, reusable utility modules, and a working end-to-end example.

📖 **Read the series:** [Medium](https://medium.com/@asidd24) · [LinkedIn Newsletter](https://www.linkedin.com/newsletters/time-series-engineered-7485187061080137728/)

---

## 📚 The Series

| # | Article | Notebook | Status |
|---|---|---|---|
| 1 | [Tabularizing Time Series: The Foundation of ML-Based Forecasting](https://medium.com/@asidd24/tabularizing-time-series-the-foundation-of-ml-based-forecasting-2070d22651ef) | [`01_tabularizing_time_series.ipynb`](notebooks/01_tabularizing_time_series.ipynb) | ✅ |
| 2 | [Why Time Series Breaks Your ML Pipeline (And How to Fix It)](https://medium.com/@asidd24/why-time-series-breaks-your-ml-pipeline-and-how-to-fix-it-3177886faa54) | [`02_ml_pipeline_leakage.ipynb`](notebooks/02_ml_pipeline_leakage.ipynb) | ✅ |
| 3 | Decomposing Time Series: Separating Signal from Noise | `03_decomposition.ipynb` | 📝 |
| 4 | Missing Data in Time Series: Beyond Simple Imputation | `04_missing_data.ipynb` | 📝 |
| 5 | Outlier Detection: Rolling Stats, LOWESS & STL | `05_outlier_detection.ipynb` | 📝 |
| 6 | Lag Features: Teaching Your Model to Remember | `06_lag_features.ipynb` | 📝 |
| 7 | Window Features: Rolling, Expanding & Exponential Smoothing | `07_window_features.ipynb` | 📝 |
| 8 | Trend Features: Making Tree-Based Models Extrapolate | `08_trend_features.ipynb` | 📝 |
| 9 | Seasonality Features: Dummies, Fourier Terms & Beyond | `09_seasonality.ipynb` | 📝 |
| 10 | Datetime & Categorical Features: The Last Mile | `10_datetime_categorical.ipynb` | 📝 |

*✅ Published · 🔜 In progress · 📝 Planned*

---

## 🎯 Who This Is For

This series is written for practitioners who work with time series data and want production-ready patterns, not textbook toy examples. Specifically:

- **Data Scientists** moving from tabular ML into forecasting or anomaly detection
- **ML Engineers** productionizing time series models and hitting leakage bugs
- **Supply Chain and Demand Planning Analysts** looking to move from Excel to Python
- **Job seekers** preparing for time series interviews at retail, CPG, energy, and pharma companies

Every notebook includes: runnable code, synthetic and real datasets, common pitfalls with fixes, and a bridge to anomaly detection where relevant.

---

## 🏗️ Repository Structure

```
time-series-feature-engineering/
├── notebooks/          # One notebook per article
├── src/                # Reusable feature engineering & validation utilities
│   ├── data.py         # Synthetic data generators + real dataset loaders
│   ├── features.py     # Lag, window, datetime, cyclical features
│   ├── validation.py   # Walk-forward CV, TimeSeriesSplit helpers
│   └── metrics.py      # MAE per horizon, custom forecasting metrics
├── tests/              # Unit tests for the src/ utilities
├── data/               # Datasets and data sourcing instructions
├── docs/               # Extended articles, glossary, references
└── images/             # Diagrams and visualizations
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Ameen2488/time-series-feature-engineering.git
cd time-series-feature-engineering
```

### 2. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter lab
```

Start with `notebooks/01_tabularizing_time_series.ipynb`.

### 4. (Optional) Install the package in editable mode

```bash
pip install -e .
```

This makes the `src/` modules importable from anywhere as `from ts_fe import features`.

---

## 📦 Key Dependencies

- **pandas ≥ 2.0** — data manipulation
- **scikit-learn ≥ 1.4** — pipelines, TimeSeriesSplit
- **lightgbm ≥ 4.0** — primary ML model
- **feature-engine ≥ 1.6** — time series feature transformers
- **skforecast ≥ 0.13** — forecasting strategy abstractions
- **statsmodels ≥ 0.14** — decomposition, ACF/PACF
- **matplotlib, seaborn** — visualization

Full list in [`requirements.txt`](requirements.txt).

---

## 🔑 Core Concepts Covered

- **Tabularization** — converting sequences into ML-consumable feature matrices
- **Temporal leakage** — the three ways future data silently contaminates training
- **Walk-forward validation** — proper cross-validation for time series
- **Forecasting strategies** — recursive vs. direct vs. multi-output
- **Feature engineering primitives** — lags, rolling windows, cyclical encoding, Fourier terms
- **Decomposition** — STL, LOWESS, classical seasonal decomposition
- **Anomaly detection** — residual-based flagging that shares feature infrastructure with forecasting

---

## 📊 Datasets Used

The notebooks use a mix of **synthetic data** (for reproducibility and pedagogical clarity) and **real public datasets**:

| Source | Domain | Notebook |
|---|---|---|
| Synthetic (generated in `src/data.py`) | General | All notebooks |
| [M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy) | Retail | 6-10 |
| [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | Retail | 4-5 |
| [ETT (Electricity Transformer)](https://github.com/zhouhaoyi/ETDataset) | Energy | 3, 7 |
| [Wikipedia Web Traffic](https://www.kaggle.com/c/web-traffic-time-series-forecasting) | Web analytics | 8-9 |

See [`data/README.md`](data/README.md) for download instructions.

---

## 🧪 Running the Tests

```bash
pytest tests/ -v
```

The test suite validates the reusable `src/` modules — particularly the feature engineering utilities that must be leakage-safe.

---

## 🤝 Contributing

If you spot a bug, have a suggestion, or want to propose an additional topic, open an issue or a pull request. Discussions are welcome via the GitHub Discussions tab.

For questions or feedback on the articles themselves, comments on the Medium posts or the LinkedIn Newsletter are the best place — I read and respond to everything there.

---

## 📖 References & Further Reading

**Books:**
- Vandeput, N. (2023). *Demand Forecasting Best Practices*. Manning.
- Vandeput, N. *Data Science for Supply Chain Forecasting*.
- Atwan, T. A. (2026). *Time Series Analysis with Python Cookbook*. Packt.
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*. [Free online](https://otexts.com/fpp3/).

**Libraries:**
- [feature-engine](https://feature-engine.trainindata.com/) — the sklearn-compatible feature engineering library
- [skforecast](https://skforecast.org/) — forecasting strategies (recursive, direct, multi-output)
- [mlforecast](https://nixtlaverse.nixtla.io/mlforecast/) — Nixtla's ML forecasting library
- [statsforecast](https://nixtlaverse.nixtla.io/statsforecast/) — Nixtla's statistical forecasting library
- [sktime](https://www.sktime.net/) — unified interface for time series ML

**Course this repo is loosely structured around:**
- [Feature Engineering for Time Series Forecasting (TrainInData)](https://github.com/trainindata/feature-engineering-for-time-series-forecasting)

---

## 👤 About the Author

I'm a Data Scientist with ~5 years of experience in time series forecasting, demand planning, and production ML. Previously at Evonik Industries (Germany), Sportradar (Germany/Switzerland), and KifferAI (Bengaluru).

This series is written from the perspective of building demand forecasting systems for real businesses — including a 500-SKU make-to-order forecasting system at Evonik that reduced planner overrides from 65% to under 30%.

- 📝 **Medium:** [medium.com/@asidd24](https://medium.com/@asidd24)
- 💼 **LinkedIn:** [linkedin.com/in/ameen-siddiqui](https://www.linkedin.com/in/ameen-siddiqui/)
- 📬 **Newsletter:** *Time Series, Engineered*

---

## 📄 License

MIT — see [LICENSE](LICENSE).

Use the code freely in your work, research, or blog posts. If you find it useful, a ⭐ on the repository and a mention in your writeup are much appreciated.

---

<p align="center"><i>If this repository saves you a bug in production, that's the goal.</i></p>
