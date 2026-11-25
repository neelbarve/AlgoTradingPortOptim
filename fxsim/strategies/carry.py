"""Trend-filtered carry trade block."""
from dataclasses import dataclass

import numpy as np

from .base import SignalStrategy, carry_diff, register

CARRY_MIN_DIFF = 0.5   # smallest rate gap (percentage points) worth trading


@dataclass(frozen=True)
class CarryTrade(SignalStrategy):
    def signal(self, frame, pair, params, rates):
        (trend,) = params
        gap = carry_diff(pair, rates)
        if abs(gap) < CARRY_MIN_DIFF:
            return np.zeros(len(frame), dtype=np.int8)
        side = 1 if gap > 0 else -1
        close = frame["close"]
        above = (close - close.rolling(trend).mean()).fillna(0.0).to_numpy()
        return np.where(side * above > 0, side, 0).astype(np.int8)


CARRY = register(CarryTrade(
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
    no_trade_hint=f"the interest-rate gap is under {CARRY_MIN_DIFF} percentage points",
    notes=(("Carry trade", "The trend filter that splits carry into separate trades is a placeholder rule."),),
    order=20,
))
