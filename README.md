# AlgoTradingPortOptim

Simulated FX strategy portfolio: trading strategies' open signals on hourly
currency data are treated as portfolio assets, and the portfolio re-optimizes
at every entry and exit.

Work in progress. This is a basic working version of the whole pipeline.

## Run

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## Data

Put the 1-hour candle files in `currencydata_raw/`, named like
`EURUSD_1h_2020-07-01_to_2025-06-30_UTC.csv` with columns
`timestamp, open, high, low, close, volume`. The folder is not in the
repository.

Pairs: EUR/USD, GBP/USD, AUD/USD, USD/CAD, USD/JPY, and EUR/JPY once its file
is added.

## Layout

| File | What it does |
|---|---|
| `app.py` | Streamlit page: one tab per pipeline stage |
| `fxsim/data.py` | Loads the candle files and aligns the pairs on one time grid |
| `fxsim/strategies.py` | SMA crossover and trend-filtered carry trade |
| `fxsim/engine.py` | Backtest engine: next-open fills, trailing stop, transaction cost, volatility-scaled slippage, carry, USD conversion |
| `fxsim/measures.py` | Return, risk and ratio measures, drawdown, trade statistics |
| `fxsim/tuning.py` | Grid search with walk-forward optimization and K-fold cross-validation |
| `fxsim/estimation.py` | Per-trade expected return and risk for each asset |
| `fxsim/optimizer.py` | Portfolio weights over open signals; efficient frontier |
| `fxsim/portfolio.py` | Margin-account simulation: rebalancing at every entry and exit, one net position per pair, trade and fill logs |

Execution defaults (0.01% transaction cost per fill, 2.5% trailing stop,
slippage proportional to hourly volatility) are in `fxsim/config.py` and can
be changed in the page's sidebar. Interest rates there are placeholders, not
data.
