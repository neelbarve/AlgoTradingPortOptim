"""MACD (moving average convergence divergence) block.

MACD line   = EMA(close, fast) - EMA(close, slow)
Signal line = EMA(MACD line, signal)
Long while the MACD line is above its signal line, short while it is below, so the position
flips at every crossover. Everything is computed from closes up to the current bar; the first
`slow + signal` bars stay flat while the averages warm up.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

from .base import SignalStrategy, register


@dataclass(frozen=True)
class MACDCrossover(SignalStrategy):
    def signal(self, frame, pair, params, rates):
        fast, slow, sig = params
        close = frame["close"]
        macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
        trigger = macd.ewm(span=sig, adjust=False).mean()
        position = np.sign((macd - trigger).to_numpy())
        position[: slow + sig] = 0
        return position.astype(np.int8)


MACD = register(MACDCrossover(
    key="macd",
    name="MACD crossover",
    short="MACD",
    description=(
        "MACD line = fast EMA minus slow EMA of the close; signal line = an EMA of the MACD line. "
        "Long while the MACD line is above the signal line, short while it is below. "
        "The classic 12/26/9 setting is tried along with slower versions for hourly bars."
    ),
    param_names=("fast", "slow", "signal"),
    grid=tuple((f, s, g) for f, s, g in product((12, 24, 48), (26, 52, 104), (9, 18)) if f < s),
    default=(12, 26, 9),
    notes=(("MACD", "Exponential averages computed from closes only (no look-ahead); flat until the "
            "averages have warmed up. Uses the shared trailing stop and costs."),),
    order=15,
))
