# AlgoTradingPortOptim

Simulated FX strategy portfolio: trading strategies' open signals on hourly
currency data are treated as portfolio assets. Exits and stop-losses close at
once; the portfolio checks whether to rebalance every 4 hours.

Work in progress. This is a basic working version of the whole pipeline.

## Run

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## Data

The data is not in this repository. Put the 1-hour candle files in
`currencydata_raw/`, named like `EURUSD_1h_2020-07-01_to_2025-06-30_UTC.csv`
with columns `timestamp, open, high, low, close, volume`.

Pairs: EUR/USD, GBP/USD, AUD/USD, USD/CAD, USD/JPY, and EUR/JPY once its file
is added.

## Pipeline

```
data -> strategy blocks -> tuning -> estimation -> portfolio (optimizer + rebalance schedule) -> results
```

Every stage is a function in `fxsim/pipeline.py`, and each replaceable piece is a parameter:

| Piece | Default | Replace by |
|---|---|---|
| Strategy | blocks in `fxsim/strategies/` | adding a module there (below) |
| Tuner | `fxsim.tuning.tune` | `run_pipeline(..., tuner=...)` |
| Estimator | `fxsim.estimation.estimate_asset` | `estimator=...` |
| Optimizer | `fxsim.optimizer.optimize_weights` | `optimizer=...` |
| Rebalance schedule | `fxsim.portfolio.clock_schedule` | `schedule=...` |

### Adding a strategy

Drop a module into `fxsim/strategies/`. It is found and registered automatically; nothing else
changes. A signal strategy only needs a position per bar:

```python
from dataclasses import dataclass
import numpy as np
from .base import SignalStrategy, register

@dataclass(frozen=True)
class Momentum(SignalStrategy):
    def signal(self, frame, pair, params, rates):
        (lookback,) = params
        return np.sign(frame["close"].pct_change(lookback).fillna(0)).to_numpy().astype(np.int8)

register(Momentum(key="mom", name="Momentum", short="Mom", description="Long after rises, short after falls.",
                  param_names=("lookback",), grid=((24,), (72,), (168,)), default=(72,)))
```

A strategy with its own state or stops subclasses `Strategy` and implements `backtest()` instead
(see `fxsim/strategies/nelstein.py`).

### Running it without the page

```python
from fxsim.data import align, build_market, load_available
from fxsim.pipeline import PipelineConfig, run_pipeline
from fxsim.config import PLACEHOLDER_RATES

raw, _ = load_available(); frames = align(raw); market = build_market(frames)
cfg = PipelineConfig(strategies=("sma", "carry", "nelstein"), pairs=tuple(frames),
                     rates=tuple(PLACEHOLDER_RATES.items()), tune_range=("2020-07-01", "2022-12-31"),
                     est_range=("2023-01-01", "2023-12-31"), sim_range=("2024-01-01", "2025-06-30"))
result = run_pipeline(market, frames, cfg)
```

## Layout

| File | What it does |
|---|---|
| `app.py` | Streamlit page: one tab per pipeline stage |
| `fxsim/data.py` | Loads the candle files and aligns the pairs on one time grid |
| `fxsim/pipeline.py` | The stages and their replaceable pieces; `run_pipeline` chains them |
| `fxsim/strategies/` | Strategy blocks (SMA crossover, MACD crossover, trend-filtered carry, Nelstein1), registered automatically |
| `fxsim/runner.py` | Backtests any registered block by key |
| `fxsim/seasonality.py` | Seasonal models per pair (hour, day, weekly cycles; SARIMA), chosen by AIC/BIC |
| `fxsim/study.py` | Nelstein1 robustness study: holdout, bias checks, benchmarks, overfitting |
| `fxsim/engine.py` | Backtest engine: next-open fills, trailing stop, transaction cost, volatility-scaled slippage, carry, USD conversion |
| `fxsim/measures.py` | Return, risk and ratio measures, drawdown, trade statistics |
| `fxsim/tuning.py` | Grid search with walk-forward optimization and K-fold cross-validation |
| `fxsim/estimation.py` | Per-trade expected return and risk for each asset |
| `fxsim/optimizer.py` | Portfolio weights over open signals; efficient frontier |
| `fxsim/portfolio.py` | Margin-account simulation: scheduled rebalance checks, immediate exits, one net position per pair, trade and fill logs |
| `What_Moves_FX.html` | Research report: which news moves FX, when, in which direction, with checks on our hourly data. Download it and open it in a browser |

The page's Methodology tab has a block diagram and every rule.

Execution defaults (0.01% transaction cost per fill, 2.5% trailing stop,
slippage proportional to hourly volatility, rebalance check every 4 hours)
are in `fxsim/config.py` and can
be changed in the page's sidebar. Interest rates there are placeholders, not
data.
