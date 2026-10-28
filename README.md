# time-series-forecasting-skill

An agent skill that teaches coding agents (Claude Code and other tools that read `SKILL.md` files) how to build sound time series forecasting pipelines in Python: from data preparation and exploration, through baselines and model selection, to evaluation, prediction intervals, and monitoring.

The skill focuses on getting the time-series-specific parts right — temporal splits, no leakage from the future, stationarity and seasonality checks, residual diagnostics, and always beating a naive baseline — rather than only fitting a model.

## What it covers

- **Data preparation**: `DatetimeIndex` with explicit frequency, explicit gaps, missing-data handling (time interpolation, forward fill, STL-based seasonal fill).
- **Exploration**: ADF + KPSS stationarity testing, differencing, classical and STL decomposition, trend/seasonal strength, ACF-based seasonality detection.
- **Models**:
  - Baselines: naive and seasonal naive
  - Univariate: AutoReg, ARIMA (ACF/PACF and AIC order selection), SARIMAX, exponential smoothing / Holt-Winters / ETS
  - Multivariate: VAR (Granger causality, impulse responses, FEVD), VARMAX, VECM, dynamic factor models
  - State space and others: unobserved components, ARDL, Markov regime switching
  - Prophet: holidays, extra regressors, built-in cross-validation
  - Machine learning: gradient boosting on lag features (global models across many series)
  - Deep learning: LSTM/GRU
- **Evaluation**: temporal train/test split, expanding-window CV (`TimeSeriesSplit`), rolling one-step-ahead evaluation, static vs dynamic forecasts, MAPE / RMSE / MAE / MASE / bias / directional accuracy / interval coverage.
- **Diagnostics**: Ljung-Box residual autocorrelation, residual plots, ARCH heteroskedasticity test.
- **Advanced**: ensemble forecasting, hierarchical forecast reconciliation (bottom-up, top-down), forecast accuracy monitoring and retraining triggers.
- **Common patterns**: sales, demand/inventory, financial prices (returns and volatility), sensor/IoT data.

## When it activates

The skill triggers on requests mentioning things like *time series*, *forecasting*, *predict future*, *trend analysis*, *seasonality*, *ARIMA*, *Prophet*, *sales forecast*, *demand prediction*, or *stock prediction*.

## Workflow the agent follows

1. Plot the series and set a regular frequency.
2. Split temporally, holding out the most recent period.
3. Explore: decomposition, stationarity tests, seasonality detection.
4. Fit naive and seasonal-naive baselines.
5. Fit candidate models suited to the data.
6. Diagnose residuals.
7. Evaluate with expanding-window CV against the baselines.
8. Forecast with prediction intervals; document choices and monitor accuracy.

A method guide maps common situations to starting models:

| Situation | Start with |
|---|---|
| Short series, single seasonality | Seasonal naive → ETS / Holt-Winters → SARIMA |
| External drivers with known future values | SARIMAX with `exog`, Prophet with regressors, ARDL |
| Daily business data, multiple seasonalities, holidays | Prophet, or STL + ETS |
| Many related series, rich features | Gradient boosting on lag features (one global model) |
| Interacting series | VAR / VECM (cointegrated) / VARMAX |
| Long, high-frequency, non-linear multivariate | LSTM/GRU (after trying the above) |

## Structure

```
.agents/skill/time-series-forecaster/
├── SKILL.md                         # Workflow, data prep, testing, CV, metrics, pitfalls
└── references/
    └── forecasting-methods.md       # How to specify, fit, forecast, and diagnose each model family
```

`SKILL.md` is loaded when the skill activates; the agent reads `references/forecasting-methods.md` on demand when choosing, fitting, or diagnosing a model.

## Installation

Copy the `time-series-forecaster` folder into your agent's skills directory. For Claude Code:

```bash
# Available in all projects
cp -r .agents/skill/time-series-forecaster ~/.claude/skills/

# Or only in one project
cp -r .agents/skill/time-series-forecaster /path/to/project/.claude/skills/
```

For other agents, place it wherever that tool discovers skills.

## Dependencies

The skill uses standard open-source Python libraries: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `prophet`, and optionally `torch` (LSTM) and `arch` (GARCH). It does not assume they are installed: the agent checks `pyproject.toml` and adds missing packages with the project's package manager (e.g. `uv add statsmodels`).

## Credits

This skill builds on the work of other skills:

- [`time-series-forecaster`](https://www.skills.sh/anton-abyzov/specweave/time-series-forecaster) from [anton-abyzov/specweave](https://github.com/anton-abyzov/specweave), the starting point for this skill's workflow, best practices, common patterns, and evaluation guidance.
- [`statsmodels` time series reference](https://github.com/tungcorn/claude-ml-skills/blob/main/scientific-skills/statsmodels/references/time_series.md) from [tungcorn/claude-ml-skills](https://github.com/tungcorn/claude-ml-skills), the source the statsmodels sections of `references/forecasting-methods.md` are adapted from.
