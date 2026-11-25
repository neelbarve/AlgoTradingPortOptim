"""Portfolio simulation: a margin account holding the strategies' open signals.

A signal's life (entry, exit, trailing stop, target) comes from the strategy's
own backtest. The portfolio decides only how much of each open signal to hold.

Exits are immediate: a strategy exit closes the position at the next bar's
open and a stop or target closes it inside the bar. Everything else waits for
the rebalance check, which runs on a fixed clock schedule (every 4 hours by
default). A check re-optimizes over the open signals when a signal opened or
closed since the last check, or when leverage has drifted above the cap. New
signals are therefore entered at the first check after they appear, if they
are still open. Rebalance trades fill at that bar's open.

The account holds one net position per pair. When several strategies trade the
same pair, only the net change is sent to market, so moving weight between
them costs nothing unless the pair's net position changes. Every fill pays a
transaction cost and volatility-scaled slippage.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PAIRS, REBALANCE_HOURS
from .data import Market
from .measures import drawdown_series
from .optimizer import frontier, optimize_weights
from .strategies import carry_diff


def clock_schedule(index: pd.DatetimeIndex, i0: int, i1: int, hours: float) -> np.ndarray:
    """Rebalance checks every `hours` of clock time, counted from midnight UTC of the first day.

    A check fires on the first bar at or after each scheduled time, so one that lands while the
    market is shut fires at the next open. With hourly bars, any interval of an hour or less
    checks on every bar.
    """
    flags = np.zeros(len(index), dtype=bool)
    if i1 <= i0:
        return flags
    start = index[i0].normalize()
    elapsed = ((index[i0:i1] - start) / pd.Timedelta(hours=1)).to_numpy()
    slot = np.floor(elapsed / float(hours))
    new = np.ones(i1 - i0, dtype=bool)
    new[1:] = slot[1:] != slot[:-1]
    flags[i0:i1] = new
    return flags


@dataclass
class Asset:
    """One strategy on one pair, with its fixed per-trade estimates."""

    key: str
    strategy: str
    pair: str
    trades: pd.DataFrame  # the strategy's trades over the simulation window
    mu: float             # per-trade expected return above cash
    sigma: float          # per-trade standard deviation


def simulate(
    market: Market,
    assets: list[Asset],
    i0: int,
    i1: int,
    *,
    capital: float,
    leverage_cap: float,
    expo_cap: float,
    objective: str,
    risk_cap: float,
    reward_floor: float,
    commission_pct: float,
    slip_vol_mult: float,
    min_trade_frac: float,
    cash_rate: float,
    rates: dict[str, float],
    rebalance_hours: float = REBALANCE_HOURS,
    optimizer=optimize_weights,
    schedule=None,
) -> dict:
    """optimizer and schedule are replaceable stages: optimizer(mu, sigma, w_cur, E, **settings) returns
    (weights, info) like optimizer.optimize_weights; schedule(index, i0, i1, hours) returns a boolean
    array marking the bars where a rebalance check happens (default: clock_schedule)."""
    A = len(assets)
    is_check = (schedule or clock_schedule)(market.index, i0, i1, rebalance_hours)
    comm = commission_pct / 100.0
    dt = market.dt_years
    index = market.index
    arr = [market.pairs[a.pair] for a in assets]
    pair_list = list(dict.fromkeys(a.pair for a in assets))
    members = [np.array([j for j, a in enumerate(assets) if a.pair == p]) for p in pair_list]
    pair_of = np.array([pair_list.index(a.pair) for a in assets])
    carry = np.array([carry_diff(a.pair, rates) / 100.0 for a in assets])
    mu = np.array([a.mu for a in assets], dtype=float)
    sigma = np.array([a.sigma for a in assets], dtype=float)
    mu = np.where(np.isfinite(mu), mu, -1.0)  # no estimate -> never held
    currencies = sorted({c for a in assets for c in PAIRS[a.pair]} - {"USD"})
    base = [PAIRS[a.pair][0] for a in assets]
    quote = [PAIRS[a.pair][1] for a in assets]

    # Index each asset's signal events by bar.
    entries: dict[int, list] = {}
    exits_open: dict[int, list] = {}
    exits_intra: dict[int, list] = {}
    exits_end: dict[int, list] = {}
    for a, asset in enumerate(assets):
        for t in asset.trades.itertuples(index=False):
            entries.setdefault(int(t.entry_i), []).append((a, int(t.direction)))
            if t.exit_reason == "signal":
                exits_open.setdefault(int(t.exit_i), []).append(a)
            elif t.exit_reason == "end":
                exits_end.setdefault(int(t.exit_i), []).append(a)
            else:
                exits_intra.setdefault(int(t.exit_i), []).append((a, float(t.exit_price), t.exit_reason))

    units = np.zeros(A)
    mark = np.zeros(A)
    dirn = np.zeros(A, dtype=int)
    active = np.zeros(A, dtype=bool)
    equity = capital

    n = i1 - i0
    eq_path = np.zeros(n)
    gross_path = np.zeros(n)
    expo_path = np.zeros((n, len(currencies)))
    pos_path = np.zeros((n, len(pair_list)))
    pair_pnl = np.zeros((n, len(pair_list)))
    totals = {"price_pnl": 0.0, "carry": 0.0, "interest": 0.0, "slippage": 0.0, "commission": 0.0}
    rebalances: list[dict] = []
    fills: list[dict] = []
    round_trips: list[dict] = []
    open_rt: list[dict | None] = [None] * A
    last_solved = None
    pending: list[str] = []

    # Per-bar P&L by asset, also credited to the asset's open round trip.
    b_price, b_carry, b_cost = np.zeros(A), np.zeros(A), np.zeros(A)

    def credit(a: int, price: float = 0.0, carry_: float = 0.0, cost: float = 0.0) -> None:
        b_price[a] += price
        b_carry[a] += carry_
        b_cost[a] += cost
        r = open_rt[a]
        if r is not None:
            r["price_pnl"] += price
            r["carry"] += carry_
            r["costs"] += cost

    def start_rt(a: int, i: int, notional: float) -> None:
        open_rt[a] = {"asset": assets[a].key, "pair": assets[a].pair,
                      "side": "long" if dirn[a] > 0 else "short", "entry_time": index[i],
                      "entry_price": arr[a].open[i], "peak_notional": notional,
                      "price_pnl": 0.0, "carry": 0.0, "costs": 0.0}

    def end_rt(a: int, i: int, price: float, reason: str) -> None:
        r = open_rt[a]
        if r is None:
            return
        r["exit_time"], r["exit_price"], r["exit_reason"] = index[i], price, reason
        r["hours_held"] = (r["exit_time"] - r["entry_time"]).total_seconds() / 3600.0
        r["net_pnl"] = r["price_pnl"] + r["carry"] - r["costs"]
        round_trips.append(r)
        open_rt[a] = None

    def exposure_matrix(ids):
        E = np.zeros((len(currencies), len(ids)))
        for j, a in enumerate(ids):
            if base[a] != "USD":
                E[currencies.index(base[a]), j] = dirn[a]
            if quote[a] != "USD":
                E[currencies.index(quote[a]), j] = -dirn[a]
        return E

    def market_fill(i: int, a: int, signed_notional: float, price: float, reason: str) -> float:
        """Send one order to market. Returns its total cost and logs the fill."""
        size = abs(signed_notional)
        s = slip_vol_mult * arr[a].vol[i] * size
        c = comm * size
        totals["slippage"] += s
        totals["commission"] += c
        fills.append({"time": index[i], "pair": assets[a].pair,
                      "side": "buy" if signed_notional > 0 else "sell", "notional": size,
                      "price": price, "slippage": s, "commission": c, "reason": reason})
        return s + c

    for i in range(i0, i1):
        b_price[:] = 0.0
        b_carry[:] = 0.0
        b_cost[:] = 0.0
        interest = 0.0
        held = np.nonzero(units)[0]

        # Carry and cash interest for the time since the previous bar.
        gross = 0.0
        if i > i0:
            for a in held:
                notional = units[a] * mark[a] * arr[a].q_usd[i - 1]
                gross += notional
                credit(a, carry_=notional * dirn[a] * carry[a] * dt[i])
            idle = max(equity - gross / leverage_cap, 0.0)
            interest = idle * cash_rate * dt[i]

        triggers, pending = pending, []
        exiting = exits_open.get(i, ())
        entering = entries.get(i, ())

        # A rebalance check happens only on the schedule, and only acts when the
        # open signals changed since the last check or leverage drifted past the cap.
        # Drift smaller than the smallest trade worth making is left alone.
        drifted = (bool(is_check[i]) and equity > 0
                   and gross / equity > leverage_cap + max(min_trade_frac, 1e-4))
        if drifted:
            triggers.append("leverage above cap")
        do_rebalance = bool(is_check[i]) and bool(triggers or exiting or entering)

        # ---- Open of the bar: signal exits (always immediate), new signals, and the rebalance if due.
        if exiting or entering or do_rebalance:
            for a in held:
                op = arr[a].open[i]
                credit(a, price=units[a] * dirn[a] * (op - mark[a]) * arr[a].q_usd[i])
                mark[a] = op
            px = np.array([p.open[i] * p.q_usd[i] for p in arr])  # USD per base unit
            before = units * dirn * px                            # signed notional per asset
            closing: dict[int, str] = {}

            for a in exiting:
                if units[a] > 0:
                    closing[a] = "signal"
                units[a], active[a] = 0.0, False
                triggers.append(f"exit {assets[a].key}")
            for a, d in entering:
                active[a], dirn[a] = True, d
                triggers.append(f"entry {assets[a].key}")

            eq_now = equity + b_price.sum() + b_carry.sum() + interest
            ids = np.nonzero(active)[0] if do_rebalance else np.array([], dtype=int)
            row = {"time": index[i], "trigger": "; ".join(triggers), "open_signals": len(ids),
                   "old_weights": "", "new_weights": "", "reward": 0.0, "risk": 0.0,
                   "unreachable": False, "note": ""}
            if len(ids) and eq_now > 0:
                cur = units[ids] * px[ids]
                w_cur = cur / eq_now
                # Opposite signals on one pair: only the side with the best estimate may be held.
                allowed = np.ones(len(ids), dtype=bool)
                for pair in {assets[a].pair for a in ids}:
                    same = [j for j, a in enumerate(ids) if assets[a].pair == pair]
                    if len({dirn[ids[j]] for j in same}) > 1:
                        keep = dirn[ids[max(same, key=lambda j: mu[ids[j]])]]
                        for j in same:
                            allowed[j] = dirn[ids[j]] == keep
                E = exposure_matrix(ids)
                cost_rate = comm + slip_vol_mult * np.array([arr[a].vol[i] for a in ids])
                w_new, info = optimizer(
                    mu[ids], sigma[ids], w_cur, E,
                    cap=leverage_cap, expo_cap=expo_cap, objective=objective,
                    risk_cap=risk_cap, reward_floor=reward_floor, slip=cost_rate, allowed=allowed,
                )
                for j, a in enumerate(ids):
                    target = w_new[j] * eq_now
                    # Resizing an existing position by less than the minimum is skipped.
                    if cur[j] > 0 and target > 0 and abs(target - cur[j]) < min_trade_frac * eq_now:
                        w_new[j] = w_cur[j]
                        continue
                    if abs(target - cur[j]) > 1e-6:
                        units[a] = target / px[a]
                        mark[a] = arr[a].open[i]
                        if target == 0.0:
                            closing[a] = "rebalanced out"
                labels = [assets[a].key for a in ids]
                row.update(
                    old_weights=", ".join(f"{l} {w:.2f}" for l, w in zip(labels, w_cur) if w > 1e-6),
                    new_weights=", ".join(f"{l} {w:.2f}" for l, w in zip(labels, w_new) if w > 1e-6),
                    reward=info["reward"], risk=info["risk"],
                    unreachable=info["unreachable"], note=info["note"],
                )
                if (w_new > 0).any():
                    last_solved = {"time": index[i], "mu": mu[ids].copy(), "sigma": sigma[ids].copy(),
                                   "E": E, "allowed": allowed, "labels": labels, "weights": w_new.copy(),
                                   "reward": info["reward"], "risk": info["risk"]}
            # One net order per pair; its cost is shared among the assets that changed.
            change = units * dirn * px - before
            # A check prompted only by drift that ends up trading nothing is not logged.
            if do_rebalance and (triggers != ["leverage above cap"] or np.abs(change).max() > 1e-6):
                rebalances.append(row)
            share = np.zeros(A)
            for k, mem in enumerate(members):
                net = change[mem].sum()
                if abs(net) < 1e-6:
                    continue
                cost = market_fill(i, int(mem[0]), net, arr[mem[0]].open[i], "rebalance")
                moved = np.abs(change[mem])
                share[mem] = cost * moved / moved.sum()

            # Round-trip bookkeeping.
            for a in range(A):
                size_before, size_after = abs(before[a]), units[a] * px[a]
                if a in closing:
                    part = share[a] * size_before / (size_before + size_after) if size_before + size_after > 0 else 0.0
                    credit(a, cost=part)
                    end_rt(a, i, arr[a].open[i], closing[a])
                    if size_after > 0:
                        start_rt(a, i, size_after)
                        credit(a, cost=share[a] - part)
                elif size_after > 0 or share[a] > 0:
                    if open_rt[a] is None and size_after > 0:
                        start_rt(a, i, size_after)
                    credit(a, cost=share[a])
                    if open_rt[a] is not None:
                        open_rt[a]["peak_notional"] = max(open_rt[a]["peak_notional"], size_after)

        # Between checks, entries and exits wait in the queue for the next one.
        if not do_rebalance:
            pending = triggers

        # ---- Inside the bar: trailing stops and targets close at once; the rest waits for the next check.
        for a, px_exit, reason in exits_intra.get(i, ()):
            qv = arr[a].q_usd[i]
            if units[a] > 0:
                credit(a, price=units[a] * dirn[a] * (px_exit - mark[a]) * qv)
                credit(a, cost=market_fill(i, a, -dirn[a] * units[a] * px_exit * qv, px_exit, reason))
                end_rt(a, i, px_exit, reason)
            units[a], active[a] = 0.0, False
            pending.append(f"{reason} {assets[a].key}")

        # ---- Close of the bar: value open positions.
        gross_close = 0.0
        for a in np.nonzero(units)[0]:
            cl, qv = arr[a].close[i], arr[a].q_usd[i]
            credit(a, price=units[a] * dirn[a] * (cl - mark[a]) * qv)
            mark[a] = cl
            notional = units[a] * cl * qv
            gross_close += notional
            pos_path[i - i0, pair_of[a]] += notional * dirn[a]
            if base[a] != "USD":
                expo_path[i - i0, currencies.index(base[a])] += notional * dirn[a]
            if quote[a] != "USD":
                expo_path[i - i0, currencies.index(quote[a])] -= notional * dirn[a]

        # Signals still open when the window ends are closed at the last close.
        for a in exits_end.get(i, ()):
            if units[a] > 0:
                cl, qv = arr[a].close[i], arr[a].q_usd[i]
                credit(a, cost=market_fill(i, a, -dirn[a] * units[a] * cl * qv, cl, "end of simulation"))
                end_rt(a, i, cl, "end of simulation")
            units[a], active[a] = 0.0, False

        equity += b_price.sum() + b_carry.sum() + interest - b_cost.sum()
        totals["price_pnl"] += b_price.sum()
        totals["carry"] += b_carry.sum()
        totals["interest"] += interest
        net_by_asset = b_price + b_carry - b_cost
        for k, mem in enumerate(members):
            pair_pnl[i - i0, k] = net_by_asset[mem].sum()
        eq_path[i - i0] = equity
        gross_path[i - i0] = gross_close
        scale = equity if equity > 0 else 1.0
        expo_path[i - i0] /= scale
        pos_path[i - i0] /= scale

    idx = index[i0:i1]
    hourly = pd.DataFrame({"equity": eq_path, "gross_notional": gross_path}, index=idx)
    hourly["ret"] = hourly["equity"].pct_change().fillna(eq_path[0] / capital - 1.0)
    hourly["leverage"] = hourly["gross_notional"] / hourly["equity"]
    hourly["drawdown"] = drawdown_series(eq_path, capital)

    front = None
    if last_solved is not None:
        m = np.where(last_solved["allowed"], last_solved["mu"], -1.0)
        risks, rewards = frontier(m, last_solved["sigma"], last_solved["E"], cap=leverage_cap, expo_cap=expo_cap)
        front = {**last_solved, "risks": risks, "rewards": rewards}

    rt_cols = ["asset", "pair", "side", "entry_time", "exit_time", "hours_held", "entry_price",
               "exit_price", "exit_reason", "peak_notional", "price_pnl", "carry", "costs", "net_pnl"]
    return {
        "hourly": hourly,
        "exposure": pd.DataFrame(expo_path, index=idx, columns=currencies),
        "positions": pd.DataFrame(pos_path, index=idx, columns=pair_list),
        "pair_pnl": pd.DataFrame(pair_pnl, index=idx, columns=pair_list).cumsum(),
        "rebalances": pd.DataFrame(rebalances),
        "fills": pd.DataFrame(fills),
        "round_trips": pd.DataFrame(round_trips, columns=rt_cols),
        "totals": {k: float(v) for k, v in totals.items()},
        "frontier": front,
        "hours_over_cap": int((hourly["leverage"] > leverage_cap * 1.0001).sum()),
    }
