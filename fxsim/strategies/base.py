"""The strategy block interface.

A strategy block knows how to backtest itself on one pair and nothing else. Tuning, estimation,
the portfolio and the page only use this interface, so any block can be swapped for another.

Two ways to write a block:
  SignalStrategy  implement signal(): a +1 / -1 / 0 position per bar, decided at the bar's close.
                  The shared engine trades it (next-open fills, trailing stop, costs, carry).
  Strategy        implement backtest() (and optionally returns()) yourself, for strategies that
                  manage their own state or stops. Return a BacktestResult like the engine does.
Call register() on an instance in the block's module; the package finds modules on its own.
"""
from dataclasses import dataclass

import numpy as np

from ..config import PAIRS
from ..engine import BacktestResult, run_backtest

REGISTRY: dict[str, "Strategy"] = {}

# Options the shared engine understands; anything else passed to a signal strategy is ignored.
ENGINE_OPTIONS = ("notional", "trail_pct", "target_pct", "commission_pct", "slip_vol_mult")


def carry_diff(pair: str, rates: dict[str, float]) -> float:
    """Base rate minus quote rate, in percent a year. What a long earns."""
    base, quote = PAIRS[pair]
    return rates[base] - rates[quote]


def _fmt(name: str, value) -> str:
    return f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}"


@dataclass(frozen=True)
class Strategy:
    key: str                          # unique id used everywhere else
    name: str                         # shown in the page
    short: str                        # used in asset labels, e.g. "SMA · EURUSD"
    description: str
    param_names: tuple[str, ...]
    grid: tuple[tuple, ...]           # parameter sets the tuner searches
    default: tuple
    per_pair: bool = False            # tuned and frozen separately for each pair
    own_stops: bool = False           # ignores the shared trailing stop and take-profit
    stop_note: str = ""               # how its own stops work (shown when own_stops)
    no_trade_hint: str = ""           # why a pair can show no trades
    notes: tuple[tuple[str, str], ...] = ()   # (topic, rule) rows for the Assumptions tab
    order: int = 100                  # position in lists

    def label(self, params: tuple) -> str:
        return ", ".join(_fmt(n, v) for n, v in zip(self.param_names, params))

    def backtest(self, market, frames, pair: str, params: tuple, rates: dict, i0: int, i1: int,
                 **options) -> BacktestResult:
        raise NotImplementedError

    def returns(self, market, frames, pair: str, params: tuple, rates: dict, i0: int, i1: int,
                **options) -> np.ndarray:
        """Hourly returns on notional. Override when it can be computed faster than a full backtest."""
        return self.backtest(market, frames, pair, params, rates, i0, i1, **options).hourly["ret"].to_numpy()


@dataclass(frozen=True)
class SignalStrategy(Strategy):
    def signal(self, frame, pair: str, params: tuple, rates: dict) -> np.ndarray:
        raise NotImplementedError

    def backtest(self, market, frames, pair, params, rates, i0, i1, **options) -> BacktestResult:
        sig = self.signal(frames[pair], pair, params, rates)
        engine_kw = {k: options[k] for k in ENGINE_OPTIONS if k in options}
        return run_backtest(market, pair, sig, i0, i1, carry_diff_pct=carry_diff(pair, rates), **engine_kw)


def register(strategy: Strategy) -> Strategy:
    if strategy.key in REGISTRY:
        raise ValueError(f"Two strategies use the key {strategy.key!r}")
    REGISTRY[strategy.key] = strategy
    return strategy
