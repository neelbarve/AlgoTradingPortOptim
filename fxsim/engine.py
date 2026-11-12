"""Backtest engine for one strategy on one pair.

Execution rules (shared by tuning, estimation and the portfolio's signals):
- A signal decided at a bar's close fills at the next bar's open.
- Trailing stop-loss: the stop sits a fixed percent behind the best price
  reached since entry (highest high for a long, lowest low for a short). The
  level used inside a bar comes from earlier bars only, so the bar's own high
  cannot rescue its low. It fills at the stop price, or at the bar's open when
  the bar gaps past it.
- A take-profit fills at its price. If one bar reaches both, the stop fills.
- After a stop or target the strategy stays flat until its signal changes.
- Every fill pays a transaction cost (percent of traded notional) and
  slippage proportional to the pair's recent hourly volatility.
- Open positions accrue carry on their USD notional for the time elapsed
  between bars, so weekends are included.
- P&L is earned in the quote currency and converted to USD at the current rate.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import COMMISSION_PCT, SLIP_VOL_MULT, TRAIL_STOP_PCT
from .data import Market

TRADE_COLUMNS = [
    "pair", "direction", "entry_i", "exit_i", "entry_time", "exit_time",
    "entry_price", "exit_price", "exit_reason", "hours_held",
    "price_pnl", "carry", "slippage", "commission", "net_pnl", "ret_gross", "ret_net",
]


@dataclass
class BacktestResult:
    trades: pd.DataFrame  # one row per trade
    hourly: pd.DataFrame  # per bar: pnl (USD, net), ret, position, stop level, equity


def run_backtest(
    market: Market,
    pair: str,
    signal: np.ndarray,
    i0: int,
    i1: int,
    *,
    notional: float = 100_000.0,
    trail_pct: float = TRAIL_STOP_PCT,
    target_pct: float = 0.0,
    commission_pct: float = COMMISSION_PCT,
    slip_vol_mult: float = SLIP_VOL_MULT,
    carry_diff_pct: float = 0.0,
) -> BacktestResult:
    """Simulate bars [i0, i1). Each trade is sized at `notional` USD."""
    p = market.pairs[pair]
    o, h, l, c, q, vol = p.open, p.high, p.low, p.close, p.q_usd, p.vol
    dt = market.dt_years
    index = market.index
    comm = commission_pct / 100.0
    trail = trail_pct / 100.0
    carry_rate = carry_diff_pct / 100.0

    n = i1 - i0
    pnl = np.zeros(n)
    held = np.zeros(n, dtype=np.int8)
    stop_path = np.full(n, np.nan)
    trades: list[dict] = []

    pos = 0          # +1 long, -1 short, 0 flat
    units = 0.0      # base-currency units held
    mark = 0.0       # last price the position was valued at
    best = 0.0       # best price since entry, for the trailing stop
    blocked = 0      # direction barred from re-entry after a stop/target
    entry_i = 0
    entry_px = tgt_px = 0.0
    t_price = t_carry = t_slip = t_comm = 0.0

    def close_trade(i: int, price: float, reason: str) -> None:
        hours = (index[i] - index[entry_i]).total_seconds() / 3600.0
        net = t_price + t_carry - t_slip - t_comm
        trades.append(
            {
                "pair": pair, "direction": pos, "entry_i": entry_i, "exit_i": i,
                "entry_time": index[entry_i], "exit_time": index[i],
                "entry_price": entry_px, "exit_price": price, "exit_reason": reason,
                "hours_held": hours, "price_pnl": t_price, "carry": t_carry,
                "slippage": t_slip, "commission": t_comm, "net_pnl": net,
                "ret_gross": (t_price + t_carry) / notional, "ret_net": net / notional,
            }
        )

    for i in range(i0, i1):
        oi, hi, li, ci, qi = o[i], h[i], l[i], c[i], q[i]
        slip = slip_vol_mult * vol[i]
        bar = 0.0

        # Carry for the time since the previous bar, on the notional held.
        if pos != 0:
            cr = units * mark * q[i - 1] * pos * carry_rate * dt[i]
            bar += cr
            t_carry += cr

        desired = int(signal[i - 1]) if i > 0 else 0
        if blocked != 0:
            if desired == blocked:
                desired = 0
            else:
                blocked = 0

        # Rule-based exit at the open.
        if pos != 0 and desired != pos:
            traded = units * oi * qi
            g = units * pos * (oi - mark) * qi
            bar += g - (slip + comm) * traded
            t_price += g
            t_slip += slip * traded
            t_comm += comm * traded
            close_trade(i, oi, "signal")
            pos = 0

        # Entry at the open.
        if pos == 0 and desired != 0:
            q_prev = q[i - 1] if i > 0 else qi
            units = notional / (oi * q_prev)
            pos = desired
            entry_i, entry_px, mark, best = i, oi, oi, oi
            tgt_px = oi * (1 + pos * target_pct / 100.0)
            traded = units * oi * qi
            bar -= (slip + comm) * traded
            t_price, t_carry, t_slip, t_comm = 0.0, 0.0, slip * traded, comm * traded

        # Trailing stop / take-profit inside the bar.
        if pos != 0:
            exit_px, reason = None, ""
            if trail > 0:
                stop_px = best * (1 - pos * trail)
                stop_path[i - i0] = stop_px
                if pos > 0 and li <= stop_px:
                    exit_px, reason = min(oi, stop_px), "trailing stop"
                elif pos < 0 and hi >= stop_px:
                    exit_px, reason = max(oi, stop_px), "trailing stop"
            if exit_px is None and target_pct > 0:
                if (pos > 0 and hi >= tgt_px) or (pos < 0 and li <= tgt_px):
                    exit_px, reason = tgt_px, "target"
            if exit_px is not None:
                traded = units * exit_px * qi
                g = units * pos * (exit_px - mark) * qi
                bar += g - (slip + comm) * traded
                t_price += g
                t_slip += slip * traded
                t_comm += comm * traded
                close_trade(i, exit_px, reason)
                blocked = pos
                pos = 0
            else:
                best = max(best, hi) if pos > 0 else min(best, li)

        # Value what is still open at the close.
        if pos != 0:
            g = units * pos * (ci - mark) * qi
            bar += g
            t_price += g
            mark = ci

        pnl[i - i0] = bar
        held[i - i0] = pos

    # Close anything still open at the end of the window.
    if pos != 0 and n > 0:
        last = i1 - 1
        traded = units * c[last] * q[last]
        cost = slip_vol_mult * vol[last] * traded
        pnl[last - i0] -= cost + comm * traded
        t_slip += cost
        t_comm += comm * traded
        close_trade(last, c[last], "end")

    hourly = pd.DataFrame(
        {"pnl": pnl, "ret": pnl / notional, "position": held, "stop": stop_path,
         "equity": notional + np.cumsum(pnl)},
        index=index[i0:i1],
    )
    return BacktestResult(trades=pd.DataFrame(trades, columns=TRADE_COLUMNS), hourly=hourly)
