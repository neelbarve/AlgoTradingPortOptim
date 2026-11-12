"""Strategies: each turns one pair's candles into a desired position per bar.

A signal value is +1 (long), -1 (short) or 0 (flat), decided when a bar closes.
The engine acts on it at the next bar's open.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PAIRS


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    short: str
    description: str
    param_names: tuple[str, ...]
    grid: tuple[tuple, ...]  # parameter sets the tuner searches
    default: tuple

    def label(self, params: tuple) -> str:
        return ", ".join(f"{n}={v}" for n, v in zip(self.param_names, params))


SMA = Strategy(
    key="sma",
    name="SMA crossover",
    short="SMA",
    description=(
        "Long while the fast moving average of the close is above the slow one, "
        "short while it is below. Flips directly between long and short."
    ),
    param_names=("fast", "slow"),
    grid=tuple((f, s) for f in (10, 20, 40) for s in (80, 120, 200)),
    default=(20, 120),
)

CARRY = Strategy(
    key="carry",
    name="Carry trade (trend-filtered)",
    short="Carry",
    description=(
        "Holds the higher-interest currency of the pair against the lower one, "
        "but only while price is on the carry side of its moving average. "
        "Flat when the rate gap is under the minimum."
    ),
    param_names=("trend",),
    grid=((50,), (100,), (200,), (400,)),
    default=(200,),
)

STRATEGIES = {s.key: s for s in (SMA, CARRY)}

# Carry trade: smallest rate gap (percentage points) worth trading.
CARRY_MIN_DIFF = 0.5


def carry_diff(pair: str, rates: dict[str, float]) -> float:
    """Base rate minus quote rate, in percent a year. What a long earns."""
    base, quote = PAIRS[pair]
    return rates[base] - rates[quote]


def make_signal(strategy_key: str, df: pd.DataFrame, pair: str, params: tuple, rates: dict) -> np.ndarray:
    close = df["close"]
    if strategy_key == "sma":
        fast, slow = params
        diff = close.rolling(fast).mean() - close.rolling(slow).mean()
        return np.sign(diff.fillna(0.0)).to_numpy().astype(np.int8)
    if strategy_key == "carry":
        (trend,) = params
        gap = carry_diff(pair, rates)
        if abs(gap) < CARRY_MIN_DIFF:
            return np.zeros(len(df), dtype=np.int8)
        side = 1 if gap > 0 else -1
        above = (close - close.rolling(trend).mean()).fillna(0.0).to_numpy()
        return np.where(side * above > 0, side, 0).astype(np.int8)
    raise KeyError(strategy_key)
