"""SMA crossover block."""
from dataclasses import dataclass

import numpy as np

from .base import SignalStrategy, register


@dataclass(frozen=True)
class SMACrossover(SignalStrategy):
    def signal(self, frame, pair, params, rates):
        fast, slow = params
        close = frame["close"]
        diff = close.rolling(fast).mean() - close.rolling(slow).mean()
        return np.sign(diff.fillna(0.0)).to_numpy().astype(np.int8)


SMA = register(SMACrossover(
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
    order=10,
))
