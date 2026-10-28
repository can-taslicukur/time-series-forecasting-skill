# Forecasting Methods Reference

How to specify, fit, forecast, and diagnose each model family: baselines, `statsmodels` (AR, ARIMA, SARIMAX, exponential smoothing/ETS, VAR/VARMAX, state space, ARDL, VECM, regime switching), Prophet, gradient boosting on lag features, and LSTM. The surrounding workflow — data preparation, stationarity testing, seasonality detection, decomposition, temporal splits, cross-validation, metrics, and pitfalls — lives in `../SKILL.md`.

statsmodels sections adapted from [tungcorn/claude-ml-skills](https://github.com/tungcorn/claude-ml-skills/blob/main/scientific-skills/statsmodels/references/time_series.md).

## Overview

- **Baselines**: naive, seasonal naive
- **Univariate statistical models**: AR, ARIMA, SARIMAX, Exponential Smoothing/ETS
- **Multivariate statistical models**: VAR, VARMAX, Dynamic Factor Models, VECM
- **State space framework**: Unobserved components, custom models, Kalman filtering
- **Prophet**: business series with multiple seasonalities, holidays, regressors
- **Machine learning**: gradient boosting on lag features
- **Deep learning**: LSTM/GRU
- **Forecasting & diagnostics**: point forecasts, prediction intervals, residual tests

## Baselines (always do this first)

```python
import pandas as pd

h = 30
m = 7  # season length
naive = pd.Series(train.iloc[-1], index=test.index[:h])
seasonal_naive = pd.Series([train.iloc[-m + (i % m)] for i in range(h)], index=test.index[:h])
```

Any model that cannot beat seasonal naive is not worth deploying.

## Univariate Statistical Models

### AutoReg (AR Model)

Autoregressive model: current value depends on past values.

**When to use:**
- Univariate time series
- Past values predict future
- Stationary series

**Model**: yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ

```python
from statsmodels.tsa.ar_model import AutoReg

# Fit AR(p) model
model = AutoReg(y, lags=5)  # AR(5)
results = model.fit()

print(results.summary())
```

**With exogenous regressors:**
```python
# AR with exogenous variables (ARX)
model = AutoReg(y, lags=5, exog=X_exog)
results = model.fit()
```

**Seasonal AR:**
```python
# Seasonal lags (e.g., monthly data with yearly seasonality)
model = AutoReg(y, lags=12, seasonal=True)
results = model.fit()
```

### ARIMA (Autoregressive Integrated Moving Average)

Combines AR, differencing (I), and MA components.

**When to use:**
- Non-stationary time series (needs differencing)
- Past values and errors predict future
- Flexible model for many time series

**Model**: ARIMA(p,d,q)
- p: AR order (lags)
- d: differencing order (to achieve stationarity)
- q: MA order (lagged forecast errors)

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA(p,d,q)
model = ARIMA(y, order=(1, 1, 1))  # ARIMA(1,1,1)
results = model.fit()

print(results.summary())
```

**Choosing p, d, q:**

1. **Determine d**: run the stationarity tests from `SKILL.md` (ADF + KPSS) and difference until the series is stationary — `d` is the number of differences needed (rarely more than 2; over-differencing hurts).

2. **Determine p and q (ACF/PACF)**:
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# After differencing to stationarity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# ACF: helps determine q (MA order)
plot_acf(y_stationary, lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')

# PACF: helps determine p (AR order)
plot_pacf(y_stationary, lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# Rules of thumb:
# - PACF cuts off at lag p → AR(p)
# - ACF cuts off at lag q → MA(q)
# - Both decay → ARMA(p,q)
```

3. **Model selection (AIC/BIC)** — grid search, or `statsforecast.models.AutoARIMA` / `pmdarima.auto_arima` if available:
```python
# Grid search for best (p,q) given d
import numpy as np

best_aic = np.inf
best_order = None

for p in range(5):
    for q in range(5):
        try:
            results = ARIMA(y, order=(p, d, q)).fit()
        except (ValueError, np.linalg.LinAlgError):
            continue  # order failed to estimate
        if results.aic < best_aic:
            best_aic = results.aic
            best_order = (p, d, q)

print(f"Best order: {best_order} with AIC: {best_aic:.2f}")
```

Information criteria (`results.aic`, `results.bic`, `results.llf`) compare models fit on the same data; confirm the winner with out-of-sample evaluation.

### SARIMAX (Seasonal ARIMA with Exogenous Variables)

Extends ARIMA with seasonality and exogenous regressors.

**When to use:**
- Seasonal patterns (monthly, quarterly data)
- External variables influence series
- Most flexible univariate model

**Model**: SARIMAX(p,d,q)(P,D,Q,s)
- (p,d,q): Non-seasonal ARIMA
- (P,D,Q,s): Seasonal ARIMA with period s — 7 daily-with-weekly, 12 monthly, 4 quarterly, 52 weekly-with-yearly

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Seasonal ARIMA for monthly data (s=12)
model = SARIMAX(y,
                order=(1, 1, 1),           # (p,d,q)
                seasonal_order=(1, 1, 1, 12))  # (P,D,Q,s)
results = model.fit(disp=False)

print(results.summary())
```

**With exogenous variables** (future values of `X_exog` are required to forecast — see Forecasting):
```python
# SARIMAX with external predictors
model = SARIMAX(y,
                exog=X_exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)
```

**Example: Monthly sales with trend and seasonality**
```python
# Typical for monthly data: (p,d,q)(P,D,Q,12)
# Start with (1,1,1)(1,1,1,12) or the "airline model" (0,1,1)(0,1,1,12)

model = SARIMAX(monthly_sales,
                order=(0, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)
```

### Exponential Smoothing

Weighted averages of past observations with exponentially decreasing weights. A strong, cheap baseline.

**When to use:**
- Simple, interpretable forecasts
- Trend and/or seasonality present
- No need for explicit model specification

**Types:**
- Simple Exponential Smoothing: no trend, no seasonality
- Holt's method: with trend
- Holt-Winters: with trend and seasonality

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Simple exponential smoothing
model = ExponentialSmoothing(y, trend=None, seasonal=None)
results = model.fit()

# Holt's method (with trend)
model = ExponentialSmoothing(y, trend='add', seasonal=None)
results = model.fit()

# Holt-Winters (trend + seasonality)
model = ExponentialSmoothing(y,
                            trend='add',           # 'add' or 'mul'
                            seasonal='add',        # 'add' or 'mul'
                            seasonal_periods=12)   # e.g., 12 for monthly
results = model.fit()
forecast = results.forecast(30)
```

**Additive vs Multiplicative:**
```python
# Additive: constant seasonal variation
# yₜ = Level + Trend + Seasonal + Error

# Multiplicative: proportional seasonal variation
# yₜ = Level × Trend × Seasonal × Error

# Choose based on data:
# - Additive: seasonal variation constant over time
# - Multiplicative: seasonal variation increases with level
```

**Innovations state space (ETS)** — use this when you need prediction intervals:
```python
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# More robust, state space formulation
model = ETSModel(y,
                error='add',           # 'add' or 'mul'
                trend='add',           # 'add', 'mul', or None
                seasonal='add',        # 'add', 'mul', or None
                seasonal_periods=12)
results = model.fit()
results.get_prediction(start=len(y), end=len(y) + 29).summary_frame(alpha=0.05)
```

## Multivariate Statistical Models

### VAR (Vector Autoregression)

System of equations where each variable depends on past values of all variables.

**When to use:**
- Multiple interrelated time series
- Bidirectional relationships
- Granger causality testing

**Model**: Each variable is AR on all variables:
- y₁ₜ = c₁ + φ₁₁y₁ₜ₋₁ + φ₁₂y₂ₜ₋₁ + ... + ε₁ₜ
- y₂ₜ = c₂ + φ₂₁y₁ₜ₋₁ + φ₂₂y₂ₜ₋₁ + ... + ε₂ₜ

```python
from statsmodels.tsa.api import VAR
import pandas as pd

# DataFrame with one column per series
wide = pd.DataFrame({'store_1': s1, 'store_2': s2, 'store_3': s3})
wide_diff = wide.diff().dropna()        # VAR needs stationary inputs

model = VAR(wide_diff)

# Select lag order using AIC/BIC/HQIC
print(model.select_order(maxlags=15).summary())

# Fit with optimal lags
results = model.fit(maxlags=15, ic='aic')
print(results.summary())

# Forecast, then undo differencing
fc_diff = results.forecast(wide_diff.to_numpy()[-results.k_ar:], steps=30)
idx = pd.date_range(wide.index[-1], periods=31, freq=wide.index.freq)[1:]
fc = pd.DataFrame(fc_diff, index=idx, columns=wide.columns).cumsum() + wide.iloc[-1]
```

If the levels are cointegrated, use VECM on levels instead (see Error Correction Models).

**Granger causality testing:**
```python
# Test if store_1 Granger-causes store_2
from statsmodels.tsa.stattools import grangercausalitytests

# Column order: [effect, cause]
test_data = wide_diff[['store_2', 'store_1']]

# Test up to max_lag
max_lag = 5
gc = grangercausalitytests(test_data, max_lag, verbose=True)

# P-values for each lag
for lag in range(1, max_lag + 1):
    p_value = gc[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: p-value = {p_value:.4f}")
```

**Impulse Response Functions (IRF):**
```python
# Trace effect of shock through system
irf = results.irf(10)  # 10 periods ahead

# Plot IRFs
irf.plot(orth=True)  # Orthogonalized (Cholesky decomposition)
plt.show()

# Cumulative effects
irf.plot_cum_effects(orth=True)
plt.show()
```

**Forecast Error Variance Decomposition:**
```python
# Contribution of each variable to forecast error variance
fevd = results.fevd(10)  # 10 periods ahead
fevd.plot()
plt.show()
```

### VARMAX (VAR with Moving Average and Exogenous Variables)

Extends VAR with MA component and external regressors.

**When to use:**
- VAR inadequate (MA component needed)
- External variables affect system
- More flexible multivariate model

```python
from statsmodels.tsa.statespace.varmax import VARMAX

# VARMAX(p, q) with exogenous variables
model = VARMAX(wide_diff,
               order=(1, 1),        # (p, q)
               exog=X_exog)
results = model.fit(disp=False)

print(results.summary())
```

### Error Correction Models (VECM)

For cointegrated series (non-stationary levels that move together).

```python
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

# Test for cointegration
johansen_test = coint_johansen(wide, det_order=0, k_ar_diff=1)

# Fit VECM on levels if cointegrated
model = VECM(wide, k_ar_diff=1, coint_rank=1)
results = model.fit()
fc = results.predict(steps=30)
```

## State Space Models

Flexible framework for custom time series models.

**When to use:**
- Custom model specification
- Unobserved components
- Kalman filtering/smoothing
- Missing data

**Unobserved components** (structural model: level/trend/seasonal as latent states; handles missing values natively):
```python
from statsmodels.tsa.statespace.structural import UnobservedComponents

model = UnobservedComponents(y, level="local linear trend", freq_seasonal=[{"period": 7, "harmonics": 3}])
results = model.fit(disp=False)
results.plot_components()
forecast = results.get_forecast(steps=30).summary_frame()
```

For fully custom state space models, subclass `statsmodels.tsa.statespace.mlemodel.MLEModel`.

**Dynamic Factor Models:**
```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# Extract common factors from multiple time series
model = DynamicFactor(wide_diff,
                      k_factors=2,          # Number of factors
                      factor_order=2)       # AR order of factors
results = model.fit(disp=False)

# Estimated factors
factors = results.factors.filtered
```

## Other statsmodels Models

### ARDL (Autoregressive Distributed Lag)

Bridges univariate and multivariate time series: y depends on its own lags and lags of X.

```python
from statsmodels.tsa.ardl import ARDL

model = ARDL(y, lags=2, exog=X, order=2)   # order = lags of each exog
results = model.fit()
```

### Regime Switching Models

For structural breaks and regime changes.

```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Markov switching model (switching mean and variance)
model = MarkovRegression(y, k_regimes=2, switching_variance=True)
results = model.fit()

# Smoothed probabilities of regimes
regime_probs = results.smoothed_marginal_probabilities
```

## Prophet

**Best for**: Business time series (sales, website traffic, user growth) with multiple seasonalities and holidays. Handles missing data and outliers reasonably well.

```python
from prophet import Prophet

pdf = train.reset_index()
pdf.columns = ["ds", "y"]

m = Prophet(seasonality_mode="multiplicative", interval_width=0.95)
m.add_country_holidays(country_name="US")
m.fit(pdf)

future = m.make_future_dataframe(periods=90, freq="D")
forecast = m.predict(future)       # columns: ds, yhat, yhat_lower, yhat_upper, trend, ...

m.plot(forecast)
m.plot_components(forecast)        # trend + seasonality + holidays
```

**Custom regressors** (values must also be supplied for the future period):

```python
m = Prophet()
m.add_regressor("marketing_spend")
m.add_regressor("temperature")
m.fit(pdf_with_regressors)         # pdf must contain both regressor columns

future = m.make_future_dataframe(periods=90)
future = future.merge(planned_regressors, on="ds", how="left")
forecast = m.predict(future)
```

**Prophet cross-validation** (built-in rolling-origin evaluation):

```python
from prophet.diagnostics import cross_validation, performance_metrics

cv = cross_validation(m, initial="730 days", period="90 days", horizon="90 days")
performance_metrics(cv)            # mape, rmse, mae, coverage per horizon
```

## Machine Learning on Lag Features

**Best for**: Many related series, rich exogenous features, non-linear effects. Often beats deep learning on tabular business data.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

def make_features(s: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=s.index)
    for lag in (1, 7, 14, 28):
        X[f"lag_{lag}"] = s.shift(lag)
    X["roll_mean_7"] = s.shift(1).rolling(7).mean()   # shift first: no leakage
    X["dow"] = s.index.dayofweek
    X["month"] = s.index.month
    return X

X = make_features(y)
data = X.assign(y=y).dropna()
train_d, test_d = data[data.index < split_date], data[data.index >= split_date]

gbm = HistGradientBoostingRegressor().fit(train_d.drop(columns="y"), train_d["y"])
pred = gbm.predict(test_d.drop(columns="y"))   # one-step-ahead
```

For multi-step horizons, either forecast recursively (feed predictions back in as lags) or train one model per horizon (direct strategy). Use `loss="quantile"` with several `quantile` values for prediction intervals.

## Deep Learning (LSTM/GRU)

**Best for**: Long, high-frequency, multivariate series with complex non-linear patterns (e.g. sensor data). Needs a lot of data; try the methods above first.

```python
import numpy as np
import torch
from torch import nn

def make_windows(arr: np.ndarray, lookback: int, horizon: int):
    X, Y = [], []
    for i in range(len(arr) - lookback - horizon + 1):
        X.append(arr[i : i + lookback])
        Y.append(arr[i + lookback : i + lookback + horizon, 0])  # target = column 0
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

# Scale with statistics from the TRAIN split only, then:
X_tr, Y_tr = make_windows(train_scaled, lookback=30, horizon=7)
model = LSTMForecaster(n_features=X_tr.shape[-1], horizon=7)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(X_tr), Y_tr)
    loss.backward()
    opt.step()
    # evaluate on a temporal validation split and stop early when it stops improving
```

Libraries such as `darts`, `neuralforecast` or `sktime` wrap this boilerplate if available.

## Forecasting with statsmodels

### Point Forecasts

```python
# Forecast h steps ahead
h = 10
forecast = results.forecast(steps=h)

# Models with exogenous variables need future exog values
forecast = results.forecast(steps=h, exog=X_future)
```

### Prediction Intervals

```python
# Get forecast with confidence intervals
forecast_df = results.get_forecast(steps=h).summary_frame(alpha=0.05)
# Contains: mean, mean_se, mean_ci_lower, mean_ci_upper

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Historical')
plt.plot(forecast_df.index, forecast_df['mean'], label='Forecast', color='red')
plt.fill_between(forecast_df.index,
                 forecast_df['mean_ci_lower'],
                 forecast_df['mean_ci_upper'],
                 alpha=0.3, color='red', label='95% CI')
plt.legend()
plt.title('Forecast with Prediction Intervals')
plt.show()
```

### Dynamic vs Static Forecasts

In-sample one-step predictions (static) look much better than true multi-step forecasts (dynamic); judge a model at the horizon you will actually forecast.

```python
# Static (one-step-ahead, using actual values)
static_forecast = results.get_prediction(start=split_point, end=len(y)-1)

# Dynamic (multi-step, using predicted values)
dynamic_forecast = results.get_prediction(start=split_point,
                                          end=len(y)-1,
                                          dynamic=True)

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))
y.plot(ax=ax, label='Actual')
static_forecast.predicted_mean.plot(ax=ax, label='Static forecast')
dynamic_forecast.predicted_mean.plot(ax=ax, label='Dynamic forecast')
ax.legend()
plt.show()
```

### Updating with New Observations

```python
# Add new data without re-estimating parameters (fast; used for rolling evaluation)
results = results.append(new_obs)        # new_obs: Series continuing y's index

# Re-estimate parameters periodically on the full history
results = SARIMAX(y_full, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
```

## Residual Diagnostics

A good model leaves white-noise residuals.

### Autocorrelation

```python
# Ljung-Box test for autocorrelation in residuals
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(results.resid, lags=[10, 20], return_df=True)
print(lb_test)
# P-values > 0.05 indicate no significant autocorrelation (good)

# Plot residual diagnostics (state space models: SARIMAX, ETS, UnobservedComponents, ...)
results.plot_diagnostics(figsize=(12, 8))
plt.show()

# Components:
# 1. Standardized residuals over time
# 2. Histogram + KDE of residuals
# 3. Q-Q plot for normality
# 4. Correlogram (ACF of residuals)
```

### Heteroskedasticity

```python
from statsmodels.stats.diagnostic import het_arch

# ARCH test for heteroskedasticity
arch_test = het_arch(results.resid, nlags=10)
print(f"ARCH test statistic: {arch_test[0]:.4f}")
print(f"p-value: {arch_test[1]:.4f}")

# If significant (p < 0.05): volatility clustering, consider a GARCH model (`arch` package)
```
