---
name: time-series-forecaster
description: Time series forecasting with ARIMA/SARIMAX, exponential smoothing, Prophet, gradient boosting on lag features, LSTM, and VAR. Activates for "time series", "forecasting", "predict future", "trend analysis", "seasonality", "ARIMA", "Prophet", "sales forecast", "demand prediction", "stock prediction". Handles trend decomposition, stationarity testing, seasonality detection, multivariate forecasting, temporal cross-validation, and confidence intervals.
---

# Time Series Forecaster

## Overview

Forecasting pipelines for time-dependent data: trend analysis, seasonality detection, and future predictions using statistical methods, machine learning, and deep learning. Uses standard open-source libraries (`pandas`, `statsmodels`, `scikit-learn`, `prophet`, optionally `torch`).

Before importing a library, check it is in `pyproject.toml`; if not, add it with the project's package manager (e.g. `uv add statsmodels`) rather than assuming it is installed.

## References

- `references/forecasting-methods.md` — all forecasting methods: naive baselines; statsmodels AutoReg, ARIMA (order selection via ACF/PACF and AIC), SARIMAX, exponential smoothing/ETS, VAR (Granger causality, impulse responses, FEVD), VARMAX, VECM, unobserved components, dynamic factor, ARDL, regime switching; Prophet (holidays, regressors, built-in CV); gradient boosting on lag features; LSTM; prediction intervals, static vs dynamic forecasts, residual and ARCH tests. Read it when choosing, fitting, or diagnosing a model.

## Workflow

1. **Plot the series** and set a regular frequency (Data Preparation).
2. **Split temporally** — hold out the most recent period.
3. **Explore**: decomposition, stationarity tests, seasonality detection.
4. **Baselines**: naive and seasonal naive (`references/forecasting-methods.md`).
5. **Candidate models** suited to the data (method guide below; code in `references/forecasting-methods.md`).
6. **Diagnose** residuals (no autocorrelation left — Ljung-Box; see reference).
7. **Evaluate** with expanding-window CV against the baselines (MASE, MAPE, interval coverage).
8. **Forecast** with prediction intervals; document choices and monitor accuracy.

**Method guide**:

| Situation | Start with |
|---|---|
| Short series, single seasonality | Seasonal naive → ETS / Holt-Winters → SARIMA |
| External drivers with known future values | SARIMAX with `exog`, Prophet with regressors, ARDL |
| Daily business data, multiple seasonalities, holidays | Prophet, or STL + ETS |
| Many related series, rich features | Gradient boosting on lag features (one global model) |
| Interacting series | VAR / VECM (cointegrated) / VARMAX |
| Long, high-frequency, non-linear multivariate | LSTM/GRU (after trying the above) |

## Why Time Series is Different

**Standard ML assumptions violated**:

- ❌ Data is NOT independent (temporal correlation)
- ❌ Data is NOT identically distributed (trends, seasonality)
- ❌ Random train/test split is WRONG (breaks temporal order)

**Time series requirements**:

- ✅ Temporal order preserved
- ✅ No data leakage from future
- ✅ Stationarity checks
- ✅ Autocorrelation analysis
- ✅ Seasonality decomposition
- ✅ Always compare against a naive baseline

## Data Preparation

Work with a `pd.Series` (or `DataFrame`) indexed by a `DatetimeIndex` with an explicit frequency:

```python
import pandas as pd

df["date"] = pd.to_datetime(df["date"])
y = df.set_index("date")["sales"].sort_index().asfreq("D")  # or "W", "MS", ...
```

`asfreq` exposes gaps as `NaN` so they can be handled deliberately (see Handling Missing Data).

## Time Series Best Practices

### 1. Temporal Train/Test Split

```python
# ❌ WRONG: Random split (data leakage!)
X_train, X_test = train_test_split(data, test_size=0.2)

# ✅ CORRECT: Temporal split
split_date = "2024-01-01"
train = data[data.index < split_date]
test = data[data.index >= split_date]

# Or use last N periods as test
train = data[:-30]  # All but last 30 observations
test = data[-30:]   # Last 30 observations
```

Fit scalers, imputers, and encoders on the train split only.

### 2. Stationarity Testing

```python
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_report(s: pd.Series) -> dict:
    s = s.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    return {
        "adf_stat": adf_stat, "adf_p": adf_p,      # H0: unit root (non-stationary)
        "kpss_stat": kpss_stat, "kpss_p": kpss_p,  # H0: stationary
        "is_stationary": adf_p < 0.05 and kpss_p > 0.05,
    }

if not stationarity_report(y)["is_stationary"]:
    y_diff = y.diff().dropna()          # first difference removes trend
    y_sdiff = y.diff(7).dropna()        # seasonal difference
```

**Stationarity Report**:

```markdown
# Stationarity Analysis

## ADF Test (Augmented Dickey-Fuller)
- Test Statistic: -2.15
- P-value: 0.23
- Critical Value (5%): -2.89
- Result: ❌ NON-STATIONARY (p > 0.05)

## Recommendation
Apply differencing (order=1) to remove trend.

After differencing:
- ADF Test Statistic: -5.42
- P-value: 0.0001
- Result: ✅ STATIONARY
```

### 3. Seasonal Decomposition

```python
import numpy as np
from statsmodels.tsa.seasonal import STL, seasonal_decompose

dec = seasonal_decompose(y, model="multiplicative", period=12)  # or "additive"
dec.plot()

stl = STL(y, period=12, robust=True).fit()  # robust to outliers, preferred
stl.plot()

# Strength of trend / seasonality (Hyndman), 0..1
trend_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))
seasonal_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))
```

Additive when the seasonal swing is constant; multiplicative when it grows with the level.

### 4. Seasonality Detection

```python
from statsmodels.tsa.stattools import acf

r = acf(y.dropna(), nlags=400)
candidates = {"weekly": 7, "monthly": 30, "yearly": 365}
threshold = 1.96 / np.sqrt(len(y.dropna()))   # approx. 95% significance band
seasonal = {name: (lag < len(r) and r[lag] > threshold) for name, lag in candidates.items()}

# Confirm with STL seasonal strength (> ~0.6 is strong) for each candidate period.
```

### 5. Cross-Validation for Time Series

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

tscv = TimeSeriesSplit(n_splits=5, test_size=30)   # expanding window
scores = []
for tr_idx, te_idx in tscv.split(y):
    tr, te = y.iloc[tr_idx], y.iloc[te_idx]
    fc = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=7).fit().forecast(len(te))
    scores.append(mean_absolute_percentage_error(te, fc))

# Plot forecast vs actual per fold and check stability of the score across folds.
```

**Rolling one-step-ahead evaluation** (realistic when the model is refreshed each period) — refitting every step is slow; update the state with new observations instead:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

res = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
preds = []
for t in range(len(test)):
    preds.append(res.forecast(1).iloc[0])
    res = res.append(test.iloc[[t]])   # adds the observation, keeps the fitted parameters
```

Evaluate at the horizon you will actually forecast — one-step-ahead scores flatter multi-step models (see "Dynamic vs Static Forecasts" in the reference). Prophet has its own rolling-origin CV (see reference).

### 6. Handling Missing Data

```python
y = y.asfreq("D")                                   # make gaps explicit

y_filled = y.interpolate(method="time", limit=3)    # max 3 consecutive gaps
y_ffill = y.ffill(limit=3)                          # e.g. for stock-like levels

# Seasonal-aware: fill from STL on a roughly interpolated series
rough = y.interpolate(method="time")
stl = STL(rough, period=7, robust=True).fit()
y_seasonal = y.fillna(stl.trend + stl.seasonal)
```

## Common Time Series Patterns

### Pattern 1: Sales Forecasting

Prophet or SARIMAX with exogenous regressors (marketing spend, price, promotions) plus holiday effects. Produce point forecasts, 80%/95% intervals, and component plots (trend, seasonality, regressors). Flag past anomalies (residuals beyond ~3σ) before training.

### Pattern 2: Demand Forecasting

Many products → one global gradient-boosting model on lag features with a `product_id` feature, or Holt-Winters per product for small catalogs. Aggregate to the planning level (daily/weekly). Use quantile forecasts to derive stockout risk and reorder points (e.g. reorder point = P90 demand over lead time).

### Pattern 3: Financial Prices

Model returns, not prices (`np.log(p).diff()`). Point forecasts are close to a random walk; the useful output is usually volatility (`arch` package, GARCH(1,1)) and risk metrics. Be explicit that predictive power is limited.

### Pattern 4: Sensor Data / IoT

High-frequency multivariate → resample (`y.resample("1h").mean()`), then VAR, gradient boosting, or LSTM. Pair the forecast with anomaly detection on residuals for maintenance alerts.

## Evaluation Metrics

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

def mase(y_true, y_pred, y_train, m: int = 1) -> float:
    """< 1 means better than the (seasonal) naive forecast."""
    scale = np.mean(np.abs(np.asarray(y_train)[m:] - np.asarray(y_train)[:-m]))
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)

def directional_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))

def interval_coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

metrics = {
    "mape": mean_absolute_percentage_error(test, pred),   # business-friendly; breaks near 0
    "rmse": root_mean_squared_error(test, pred),          # penalizes large errors
    "mae": mean_absolute_error(test, pred),               # robust to outliers
    "mase": mase(test, pred, train, m=7),                 # scale-independent
    "bias": float(np.mean(np.asarray(pred) - np.asarray(test))),
    "directional_accuracy": directional_accuracy(test, pred),
}
```

**Evaluation Report**:

```markdown
# Time Series Forecast Evaluation

## Point Metrics
- MAPE: 8.2% (target: <10%) ✅
- RMSE: 124.5
- MAE: 98.3
- MASE: 0.85 (< 1 = better than naive forecast) ✅

## Directional Accuracy
- Correct direction: 73% (up/down predictions)

## Forecast Bias
- Mean Error: -5.2 (slight under-forecasting)
- Bias: -2.1%

## Confidence Intervals
- 80% interval coverage: 79.2% ✅
- 95% interval coverage: 94.1% ✅

## Recommendation
✅ DEPLOY: Model meets accuracy targets and is well-calibrated.
```

## Advanced Features

### 1. Ensemble Forecasting

```python
forecasts = {"sarimax": sarimax_fc, "prophet": prophet_fc, "gbm": gbm_fc}  # aligned pd.Series
weights = {"sarimax": 0.3, "prophet": 0.5, "gbm": 0.2}                     # e.g. inverse CV error
ensemble = sum(weights[k] * forecasts[k] for k in forecasts)
```

Simple averages of diverse models are often more accurate and more stable than any single model.

### 2. Forecast Reconciliation

For hierarchical series (total = store1 + store2 + store3), forecasts must add up:

```python
# Bottom-up: forecast the leaves, sum to the total
total_fc = store1_fc + store2_fc + store3_fc

# Top-down: forecast the total, split by historical proportions
shares = wide.tail(90).sum() / wide.tail(90).sum().sum()
leaf_fc = pd.DataFrame({c: total_fc * shares[c] for c in wide.columns})
```

The `hierarchicalforecast` package provides MinT and other optimal reconciliation methods.

### 3. Forecast Monitoring

```python
log = pd.DataFrame({"forecast": past_forecasts, "actual": actual_values})
log["ape"] = (log["forecast"] - log["actual"]).abs() / log["actual"].abs()
rolling_mape = log["ape"].rolling(28).mean()

baseline_mape = log["ape"].iloc[:90].mean()
if rolling_mape.iloc[-1] > 1.15 * baseline_mape:
    print("⚠️ Forecast accuracy degraded by >15% — retrain the model")
```

## Common Pitfalls

1. **Data leakage**: random splits, scalers/imputers fit on all data, rolling features without `shift(1)`.
2. **Not checking stationarity** before ARIMA/VAR, or over-differencing a stationary series.
3. **Wrong seasonal period**: `m=12` monthly, `4` quarterly, `7` daily-weekly.
4. **Overfitting**: too many parameters for the data length; prefer simpler models and compare AIC/BIC plus out-of-sample error.
5. **Ignoring residual autocorrelation**: the model has missed structure.
6. **MAPE with zeros or near-zero values**: use MAE, MASE, or WAPE instead.
7. **Exogenous variables without future values**: SARIMAX/Prophet regressors need values for the forecast period — forecasting them too adds error.
8. **Evaluating one-step-ahead when you deploy multi-step forecasts.**
9. **No baseline comparison** and **no prediction intervals**.
10. **Untracked transformations**: document log/differencing/scaling and invert them on the forecasts.
11. **Unhandled missing data**: gaps silently break lags, seasonality, and model estimation.

## Summary

Time series forecasting requires specialized techniques:

- ✅ Temporal validation (no random split)
- ✅ Naive baseline comparison
- ✅ Stationarity testing
- ✅ Seasonality detection
- ✅ Trend decomposition
- ✅ Cross-validation (expanding window)
- ✅ Confidence intervals
- ✅ Forecast monitoring

Document the chosen method, the alternatives tried with their CV scores, the detected seasonalities, and the forecast horizon alongside the model so forecasts are reproducible.
