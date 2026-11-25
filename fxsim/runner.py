"""One entry point for backtesting any registered strategy block on one pair.

The runner never names a strategy: it looks the key up in the registry and calls the block's
own backtest() or returns(). Options a block does not use are ignored by it.
"""
import numpy as np
import pandas as pd

from .data import Market
from .engine import BacktestResult
from .strategies import STRATEGIES


def backtest(market: Market, frames: dict[str, pd.DataFrame], key: str, pair: str, params: tuple,
             rates: dict[str, float], i0: int, i1: int, **options) -> BacktestResult:
    """Backtest one strategy on one pair over bars [i0, i1).

    Shared execution options: notional, trail_pct, target_pct, commission_pct, slip_vol_mult.
    A block may accept more (Nelstein1: mode, s1_mode, delay, trend_k).
    """
    return STRATEGIES[key].backtest(market, frames, pair, params, dict(rates), i0, i1, **options)


def backtest_returns(market: Market, frames, key, pair, params, rates, i0, i1, **options) -> np.ndarray:
    """Hourly returns of `backtest`, using the block's faster path when it has one."""
    return STRATEGIES[key].returns(market, frames, pair, params, dict(rates), i0, i1, **options)
