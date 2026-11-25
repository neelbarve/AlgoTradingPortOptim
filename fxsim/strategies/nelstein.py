"""Nelstein1: a regime-switching strategy (range breakout that hands over to trend following).

States
  S1 (range)  The last x bars are not trending. A band is drawn around the x-bar mean of the
              close at y times the average reversion (mean absolute deviation of the closes
              from that mean). A close above the band goes long, below it goes short, with a
              z% trailing stop.
  S2 (trend)  The net move over the last x bars is large compared with a random walk.
              Positions follow the trend with a trailing stop at w times the average pullback
              inside the window (the temporary dips in an uptrend, rallies in a downtrend).

Transitions, decided at each bar's close and filled at the next bar's open
  flat, trending                  -> enter S2 in the trend direction, unless a new range
                                     must form first (see below)
  flat, range, close beyond band  -> enter S1 in the breakout direction
  S1 position, trend the same way -> becomes S2: the stop switches to w x average pullback
  any position, trend reverses    -> close and enter S2 the other way
  stopped out                     -> flat; no direct S2 entry until a range has formed again
                                     (the "new S1 state"), so a stop is not bought straight back

Regime test
  Efficiency ratio ER = |c_t - c_{t-x}| / sum |c_k - c_{k-1}| over the window. Under a random
  walk ER shrinks like 1/sqrt(x), so a fixed cut-off would call most short windows trending and
  almost no long ones. The test is therefore ER * sqrt(x) >= TREND_K: the net move must be
  TREND_K times what a random walk of the same choppiness would produce.

Timing (no look-ahead)
  Every input at bar t uses closes up to t. The band at bar t comes from bars t-x .. t-1, so a
  bar never defines the band it breaks. Stops inside a bar use the best price of earlier bars.
"""
import numpy as np
import pandas as pd

from dataclasses import dataclass
from itertools import product

from ..engine import TRADE_COLUMNS, BacktestResult
from .base import Strategy, carry_diff, register

try:  # compiled when numba is installed; the same code runs as plain Python otherwise
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

TREND_K = 1.75          # trend when the net move is 1.75x a random walk's (about 20% of bars)
STOP_FLOOR = 0.0005     # smallest stop distance (0.05%), so a quiet window cannot set a zero stop
STATE_NAMES = {1: "S1", 2: "S2"}
# Direction choices and the (mode, s1_mode) each one runs with. A Nelstein1 parameter tuple may
# carry one of these names as a fifth element, e.g. (72, 2.0, 4, 1.0, "Reversed").
DIRECTIONS = {"As specified": (1, None), "Reversed": (-1, None), "S1 faded": (1, -1)}
REASONS = {0: "signal", 1: "trailing stop", 2: "end"}

_FEATURE_CACHE: dict = {}


@njit(cache=True)
def _window_stats(c, x):
    """For each window of x closes ending at bar e: mean, mean absolute deviation, and the
    average dip below the running high / rally above the running low, as fractions."""
    n = len(c)
    mean = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    pbu = np.full(n, np.nan)
    pbd = np.full(n, np.nan)
    for e in range(x - 1, n):
        j = e - x + 1
        s = 0.0
        for k in range(j, e + 1):
            s += c[k]
        m = s / x
        dev = 0.0
        hi = c[j]
        lo = c[j]
        su = 0.0
        sd = 0.0
        for k in range(j, e + 1):
            v = c[k]
            dev += abs(v - m)
            if v > hi:
                hi = v
            if v < lo:
                lo = v
            su += (hi - v) / hi
            sd += (v - lo) / lo
        mean[e] = m
        mad[e] = dev / x
        pbu[e] = su / x
        pbd[e] = sd / x
    return mean, mad, pbu, pbd


def compute_features(close: np.ndarray, x: int, trend_k: float = TREND_K):
    """Per-bar inputs for lookback x: trend (+1/-1/0), band mean and width, pullbacks."""
    c = np.ascontiguousarray(close, dtype=float)
    n = len(c)
    trend = np.zeros(n, dtype=np.int8)
    bmean = np.full(n, np.nan)
    bmad = np.full(n, np.nan)
    if n <= x:
        return trend, bmean, bmad, np.full(n, np.nan), np.full(n, np.nan)

    absd = np.abs(np.diff(c, prepend=c[0]))
    cs = np.cumsum(absd)
    path = np.full(n, np.nan)
    path[x:] = cs[x:] - cs[:-x]
    net = np.full(n, np.nan)
    net[x:] = c[x:] - c[:-x]
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.abs(net) / path * np.sqrt(x)
    trend[:] = np.where(score >= trend_k, np.sign(net), 0)

    mean, mad, pbu, pbd = _window_stats(c, int(x))
    bmean[1:] = mean[:-1]   # the band at bar t comes from the window ending at t-1
    bmad[1:] = mad[:-1]
    return trend, bmean, bmad, pbu, pbd


def features_cached(close: np.ndarray, x: int, trend_k: float = TREND_K):
    """compute_features, remembered for arrays that live for the whole session (the market)."""
    key = (id(close), int(x), float(trend_k))
    hit = _FEATURE_CACHE.get(key)
    if hit is not None and hit[0] is close:
        return hit[1]
    feats = compute_features(close, x, trend_k)
    _FEATURE_CACHE[key] = (close, feats)  # holding the array keeps its id from being reused
    return feats


@njit(cache=True)
def _record(T, k, d, ei, xi, epx, xpx, reason, tp, tc, ts, tm, est, s2):
    T[k, 0] = d
    T[k, 1] = ei
    T[k, 2] = xi
    T[k, 3] = epx
    T[k, 4] = xpx
    T[k, 5] = reason
    T[k, 6] = tp
    T[k, 7] = tc
    T[k, 8] = ts
    T[k, 9] = tm
    T[k, 10] = est
    T[k, 11] = s2


@njit(cache=True)
def _core(o, h, l, c, q, vol, dt, trend, bmean, bmad, pbu, pbd,
          i0, i1, y, w, z, mode, s1_mode, delay, notional, comm, slipm, carry_rate, stop_floor):
    n = i1 - i0
    pnl = np.zeros(n)
    held = np.zeros(n, dtype=np.int8)
    state = np.zeros(n, dtype=np.int8)
    stop_path = np.full(n, np.nan)
    T = np.zeros((n + 1, 12))
    nt = 0

    pos = 0
    st = 0
    units = 0.0
    mark = 0.0
    best = 0.0
    stopd = 0.0
    p_act = 0
    p_dir = 0
    p_state = 0
    p_stopd = 0.0
    need_range = False
    entry_i = 0
    entry_px = 0.0
    entry_state = 0
    s2 = 0
    tp = 0.0
    tc = 0.0
    ts = 0.0
    tm = 0.0

    for i in range(i0, i1):
        oi = o[i]
        hi = h[i]
        li = l[i]
        ci = c[i]
        qi = q[i]
        slip = slipm * vol[i]
        bar = 0.0

        # Carry for the time since the previous bar.
        if pos != 0:
            cr = units * mark * q[i - 1] * pos * carry_rate * dt[i]
            bar += cr
            tc += cr

        # Orders decided at the previous close fill at this open.
        if p_act != 0:
            if pos != 0:
                traded = units * oi * qi
                g = units * pos * (oi - mark) * qi
                bar += g - (slip + comm) * traded
                tp += g
                ts += slip * traded
                tm += comm * traded
                _record(T, nt, pos, entry_i, i, entry_px, oi, 0, tp, tc, ts, tm, entry_state, s2)
                nt += 1
                pos = 0
            q_prev = q[i - 1] if i > 0 else qi
            units = notional / (oi * q_prev)
            pos = p_dir
            st = p_state
            stopd = p_stopd
            entry_i = i
            entry_px = oi
            mark = oi
            best = oi
            entry_state = p_state
            s2 = 1 if p_state == 2 else 0
            traded = units * oi * qi
            bar -= (slip + comm) * traded
            tp = 0.0
            tc = 0.0
            ts = slip * traded
            tm = comm * traded
            p_act = 0

        # Trailing stop inside the bar, from the best price of earlier bars.
        if pos != 0:
            spx = best * (1.0 - pos * stopd)
            stop_path[i - i0] = spx
            hit = False
            xpx = 0.0
            if pos > 0 and li <= spx:
                hit = True
                xpx = min(oi, spx)
            elif pos < 0 and hi >= spx:
                hit = True
                xpx = max(oi, spx)
            if hit:
                traded = units * xpx * qi
                g = units * pos * (xpx - mark) * qi
                bar += g - (slip + comm) * traded
                tp += g
                ts += slip * traded
                tm += comm * traded
                _record(T, nt, pos, entry_i, i, entry_px, xpx, 1, tp, tc, ts, tm, entry_state, s2)
                nt += 1
                pos = 0
                st = 0
                need_range = True
            elif pos > 0:
                best = max(best, hi)
            else:
                best = min(best, li)

        # Value what is still open at the close.
        if pos != 0:
            g = units * pos * (ci - mark) * qi
            bar += g
            tp += g
            mark = ci

        # Decide at the close; acted on at the next open.
        j = i - delay
        if j >= 0 and i < i1 - 1:
            raw = trend[j]
            tr = raw * mode
            pb = pbu[j] if raw > 0 else pbd[j]
            if not (pb == pb):
                pb = 0.0
            sd2 = max(w * pb, stop_floor)
            if raw == 0:
                need_range = False
            if pos == 0:
                if tr != 0:
                    if not need_range:
                        p_act = 1
                        p_dir = tr
                        p_state = 2
                        p_stopd = sd2
                else:
                    upper = bmean[j] + y * bmad[j]
                    lower = bmean[j] - y * bmad[j]
                    if c[j] > upper:
                        p_act = 1
                        p_dir = s1_mode
                        p_state = 1
                        p_stopd = z
                    elif c[j] < lower:
                        p_act = 1
                        p_dir = -s1_mode
                        p_state = 1
                        p_stopd = z
            elif tr == pos and st == 1:
                st = 2
                stopd = sd2
                s2 = 1
            elif tr == -pos:
                p_act = 2
                p_dir = tr
                p_state = 2
                p_stopd = sd2

        pnl[i - i0] = bar
        held[i - i0] = pos
        state[i - i0] = st

    # Close anything still open at the end of the window.
    if pos != 0 and n > 0:
        last = i1 - 1
        traded = units * c[last] * q[last]
        cost = (slipm * vol[last] + comm) * traded
        pnl[last - i0] -= cost
        ts += slipm * vol[last] * traded
        tm += comm * traded
        _record(T, nt, pos, entry_i, last, entry_px, c[last], 2, tp, tc, ts, tm, entry_state, s2)
        nt += 1
    return pnl, held, state, stop_path, T[:nt]


def nelstein_returns(pa, dt, i0, i1, params, *, notional=100_000.0, commission_pct=0.01, slip_vol_mult=0.05,
                     carry_diff_pct=0.0, mode=1, s1_mode=None, delay=0, trend_k=TREND_K, cache=True) -> np.ndarray:
    """Hourly returns only (no trade list), for grid searches and benchmarks."""
    x, y, w, z = params
    feats = (features_cached if cache else compute_features)(pa.close, int(x), trend_k)
    pnl = _core(
        pa.open, pa.high, pa.low, pa.close, pa.q_usd, pa.vol, dt, *feats,
        int(i0), int(i1), float(y), float(w), float(z) / 100.0, int(mode),
        int(mode if s1_mode is None else s1_mode), int(delay),
        float(notional), commission_pct / 100.0, float(slip_vol_mult), carry_diff_pct / 100.0, STOP_FLOOR,
    )[0]
    return pnl / notional


def run_nelstein(
    pa,
    dt: np.ndarray,
    index: pd.DatetimeIndex,
    i0: int,
    i1: int,
    params: tuple,
    *,
    notional: float = 100_000.0,
    commission_pct: float = 0.01,
    slip_vol_mult: float = 0.05,
    carry_diff_pct: float = 0.0,
    mode: int = 1,
    s1_mode: int | None = None,
    delay: int = 0,
    trend_k: float = TREND_K,
    pair: str = "",
    cache: bool = True,
) -> BacktestResult:
    """Backtest Nelstein1 on bars [i0, i1). params = (x bars, y band, w pullback, z trailing %).

    mode=-1 reverses every directional decision (fades breakouts and trends). s1_mode overrides
    the direction of S1 breakout entries only (-1 fades them and keeps trend following).
    delay adds bars between a decision's information and its fill, for the look-ahead check.
    """
    x, y, w, z = params
    feats = (features_cached if cache else compute_features)(pa.close, int(x), trend_k)
    pnl, held, state, stop_path, T = _core(
        pa.open, pa.high, pa.low, pa.close, pa.q_usd, pa.vol, dt, *feats,
        int(i0), int(i1), float(y), float(w), float(z) / 100.0, int(mode),
        int(mode if s1_mode is None else s1_mode), int(delay),
        float(notional), commission_pct / 100.0, float(slip_vol_mult), carry_diff_pct / 100.0, STOP_FLOOR,
    )

    if len(T):
        ei = T[:, 1].astype(int)
        xi = T[:, 2].astype(int)
        net = T[:, 6] + T[:, 7] - T[:, 8] - T[:, 9]
        trades = pd.DataFrame({
            "pair": pair,
            "direction": T[:, 0].astype(int),
            "entry_i": ei,
            "exit_i": xi,
            "entry_time": index[ei],
            "exit_time": index[xi],
            "entry_price": T[:, 3],
            "exit_price": T[:, 4],
            "exit_reason": [REASONS[int(r)] for r in T[:, 5]],
            "hours_held": (index[xi] - index[ei]).total_seconds() / 3600.0,
            "price_pnl": T[:, 6],
            "carry": T[:, 7],
            "slippage": T[:, 8],
            "commission": T[:, 9],
            "net_pnl": net,
            "ret_gross": (T[:, 6] + T[:, 7]) / notional,
            "ret_net": net / notional,
            "entry_state": [STATE_NAMES[int(s)] for s in T[:, 10]],
            "reached_s2": T[:, 11] > 0.5,
        })
    else:
        trades = pd.DataFrame(columns=TRADE_COLUMNS + ["entry_state", "reached_s2"])

    hourly = pd.DataFrame(
        {"pnl": pnl, "ret": pnl / notional, "position": held, "stop": stop_path,
         "equity": notional + np.cumsum(pnl), "state": state},
        index=index[i0:i1],
    )
    return BacktestResult(trades=trades, hourly=hourly)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------
NELSTEIN_OPTIONS = ("notional", "commission_pct", "slip_vol_mult", "mode", "s1_mode", "delay", "trend_k")


@dataclass(frozen=True)
class Nelstein1(Strategy):
    """Parameters are (x, y, w, z), optionally followed by a direction name from DIRECTIONS."""

    def _prepare(self, params: tuple, options: dict) -> tuple[tuple, dict]:
        opts = {k: options[k] for k in NELSTEIN_OPTIONS if k in options}
        if len(params) > 4:
            opts["mode"], opts["s1_mode"] = DIRECTIONS[params[4]]
        return tuple(params[:4]), opts

    def label(self, params: tuple) -> str:
        text = super().label(params[:4])
        return text + (f", {params[4].lower()}" if len(params) > 4 else "")

    def backtest(self, market, frames, pair, params, rates, i0, i1, **options) -> BacktestResult:
        p, opts = self._prepare(params, options)
        return run_nelstein(market.pairs[pair], market.dt_years, market.index, i0, i1, p,
                            carry_diff_pct=carry_diff(pair, rates), pair=pair, **opts)

    def returns(self, market, frames, pair, params, rates, i0, i1, **options) -> np.ndarray:
        p, opts = self._prepare(params, options)
        return nelstein_returns(market.pairs[pair], market.dt_years, i0, i1, p,
                                carry_diff_pct=carry_diff(pair, rates), **opts)


NELSTEIN = register(Nelstein1(
    key="nelstein",
    name="Nelstein1",
    short="Nel1",
    description=(
        "Regime-switching. If the last x bars are ranging (S1), it waits for a close beyond a band "
        "of y times the average reversion around the x-bar mean, then trades the breakout with a "
        "z% trailing stop. If they are trending (S2), it follows the trend with a trailing stop at "
        "w times the average pullback. An S1 trade that starts trending moves into S2. "
        "Tuned separately for each pair."
    ),
    param_names=("x", "y", "w", "z"),
    grid=tuple(product((24, 72, 168), (1.5, 2.0, 3.0), (2, 4, 8), (0.5, 1.0, 2.0))),
    default=(72, 2.0, 4, 1.0),
    per_pair=True,
    own_stops=True,
    stop_note=("its own stops: a z% trailing stop on an S1 breakout, switching to w × the window's "
               "average pullback once the trade starts trending (S2). After a stop, no trend entry is "
               "taken until a new range has formed"),
    no_trade_hint="no breakout or trend signal fired in this period",
    notes=(
        ("Nelstein1 regime test", f"Trending when the efficiency ratio × √x is at least {TREND_K:g} (the net "
         "move is that many times a random walk's). Fixed, not tuned."),
        ("Nelstein1 measures", "Average reversion = mean absolute deviation of closes from the x-bar mean. "
         "Average pullback = mean dip below the running high (uptrend) or rally above the running low "
         "(downtrend) inside the window. Trend stop floor 0.05%."),
        ("Nelstein1 after a stop", "No direct trend entry until a range has formed again, so a stopped "
         "trend is not bought straight back. The sidebar trailing stop and take-profit do not apply."),
    ),
    order=30,
))
