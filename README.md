# Cross-Sectional ML Momentum Strategy: QuantQuest

**QuantQuest, Xpecto (E-Summit '26), IIT Mandi** · March 2026 · solo submission · **3rd Place**

Notebook: [`TanmayPratapSingh_QuantQuest_ESummit26.ipynb`](TanmayPratapSingh_QuantQuest_ESummit26.ipynb). Report: [`TanmayPratapSingh_QuantQuest_ESummit26.pdf`](TanmayPratapSingh_QuantQuest_ESummit26.pdf).

## Problem

Each week, the task is to choose which stocks to hold from a fixed universe of 10 US large caps. The competition fixed these parameters:

- **Universe:** AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, V, JNJ and BRK-B.
- **Data:** daily OHLCV from 2017-01-01 to 2025-01-01. Train ends 2022-12-31 and test starts 2023-01-01.
- **Portfolio:** long-only, holding the top 2 stocks, rebalanced weekly.
- **Costs:** 0.2% round-trip, deducted every week.

## Data

- Daily OHLCV is downloaded from Yahoo Finance (`yfinance`, `auto_adjust=True`) for the 10 stocks, plus SPY and QQQ as regime proxies.
- That gives 2,012 rows per ticker, with no nulls.
- The weekly snapshot (last trading day of each week) has 3,680 rows: 2,630 for training (2017-12-22 to 2022-12-30) and 1,050 for testing (2023-01-06 to 2024-12-31, 104 weeks).

## Method

**Features (43).**

- 29 are cross-sectional percentile ranks within the 10-stock universe:
  - multi-horizon returns from 3 to 252 days
  - momentum acceleration
  - distance to the 52-week high and the Donchian channel
  - volatility, including Parkinson
  - RSI, MACD, Bollinger and ADX
  - volume ratio and windowed OBV
  - moving-average ratios
  - alpha against SPY and QQQ
- 14 are raw SPY and QQQ regime features: 5-, 20- and 60-day returns, 20-day volatility, a volatility regime, a trend flag and a bull flag.
- Ranking across the universe turns absolute indicator levels into comparisons between peers.

**Target.** The model predicts whether a stock's 5-day forward return is positive.

**Models.** A 3-model ensemble:

- Logistic Regression (C = 1, class-balanced).
- ExtraTrees with manual Platt calibration.
- LightGBM, heavily regularised.

The ensemble weights adapt to model health. In the final run ExtraTrees degenerated (its probability std was 0.017), so the ensemble fell back to LR 55% and LGB 45%.

**Portfolio.** Each week the strategy ranks all 10 stocks by ensemble probability, holds the top 2 with probability-weighted allocation, and also reports an equal-weight variant.

**Leakage controls.**

- The train/test split is temporal.
- RobustScaler is fit on the training data only.
- `TimeSeriesSplit` (3 folds) inside the training period is used for Platt calibration and LightGBM early stopping.
- OBV is computed over a window rather than cumulatively.

**Walk-forward variant.** An expanding window is retrained every 6 months across the test period.

## Results

Test period is 2023-01 to 2025-01, 104 weeks.

| | Prob-weighted gross | Prob-weighted net | Equal-weight net | Benchmark: EW all 10 (no TC) |
|---|---|---|---|---|
| Total return | 110.73% | **71.35%** | 71.66% | 110.21% |
| Annualised return | 50.12% | 35.39% | 35.52% | 46.81% |
| Annualised volatility | 26.61% | 26.61% | 26.65% | 16.01% |
| Sharpe | 1.533 | **1.142** | 1.144 | 2.408 |
| Sortino | 3.395 | 2.382 | 2.403 | 5.052 |
| Max drawdown | −16.74% | −19.23% | −19.18% | −9.23% |
| Hit rate | 58.65% | 53.85% | 53.85% | 64.42% |

| Walk-forward (6-month expanding retrain) | Value |
|---|---|
| Net total return | 40.02% |
| Sharpe | 0.80 |
| Max drawdown | −15.63% |
| Weeks | 100 |

**Model diagnostics (test AUC; train AUC in brackets)**

| Model | Test AUC | Train AUC |
|---|---|---|
| Ensemble | 0.5382 | 0.6662 |
| Logistic Regression | 0.5396 | 0.5876 |
| LightGBM | 0.5151 | 0.7416 |
| ExtraTrees | 0.5109 | 0.6772 |

**What the results do and do not show**

- The strategy **underperforms** a simple equal-weight hold of all 10 stocks, on both return (71.35% vs 110.21%) and Sharpe (1.142 vs 2.408). The comparison also favours the benchmark, which is charged no transaction costs.
- Probability weighting adds nothing over equal weighting (71.35% vs 71.66%).
- The predictive signal is weak: the test AUC is 0.538 on 1,050 stock-weeks. Tree models show large train/test gaps.
- The walk-forward result (net 40.02%, Sharpe 0.80) is weaker than the static model.
- Several design iterations were evaluated on the same test window before the final configuration was chosen. The test period is therefore not a pristine holdout.
- Costs are charged at 0.2% every week, whatever the turnover.

## How to run

The notebook installs its dependencies in its first code cell and downloads all data from Yahoo Finance, so no data files are needed.

```bash
pip install yfinance ta lightgbm scikit-learn pandas numpy matplotlib seaborn scipy jupyter
jupyter notebook TanmayPratapSingh_QuantQuest_ESummit26.ipynb   # Run All
```

Running it produces `weekly_predictions.csv` (1,050 rows), `portfolio_selections.csv` (104 rows), `backtest_returns.csv` (104 rows), `performance_report.csv` and `strategy_dashboard.png`. Yahoo Finance revises adjusted prices over time, so a re-run may differ slightly from the numbers above.

## Repository layout

```
TanmayPratapSingh_QuantQuest_ESummit26.ipynb   full pipeline: data -> features -> models -> backtest -> walk-forward -> exports
TanmayPratapSingh_QuantQuest_ESummit26.pdf     submitted research report
```
