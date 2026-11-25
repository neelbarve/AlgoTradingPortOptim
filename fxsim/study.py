"""Robustness study for Nelstein1: performance, biases, overfitting and benchmarks.

Every Sharpe ratio here is on the trade notional with no cash rate subtracted, so zero means
no edge. Parameters are tuned per pair on the tuning period and judged on the later
simulation period ("holdout"), which tuning never sees.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import PAIRS
from .data import Market, PairArrays, trailing_vol
from .measures import compute_measures, trade_stats
from .strategies.nelstein import DIRECTIONS, NELSTEIN, TREND_K, features_cached, nelstein_returns
from .runner import backtest, backtest_returns
from .strategies import carry_diff
from .tuning import tune

KEY = NELSTEIN.key

N_SYNTHETIC = 30
BLOCK = 24
WARMUP = 200


def sharpe(ret) -> float:
    return compute_measures(np.asarray(ret, dtype=float))["sharpe"]


def _metrics(res) -> dict:
    m = compute_measures(res.hourly["ret"].to_numpy())
    s = trade_stats(res.trades["net_pnl"].to_numpy(), res.trades["hours_held"].to_numpy())
    return {"sharpe": m["sharpe"], "total_return": m["total_return"], "max_drawdown": m["max_drawdown"],
            "trades": s["trades"], "win_rate": s["win_rate"], "profit_factor": s["profit_factor"]}


def synthetic_pair(market: Market, pair: str, i0: int, i1: int, seed: int) -> tuple[PairArrays, np.ndarray]:
    """A random-walk version of the pair: whole bars resampled in 24-bar blocks.

    Each bar keeps its open/high/low/close shape relative to the previous close, and blocks keep
    intraday volatility patterns, but the order of blocks is random, so any trend or reversion
    structure across days is destroyed. Returns arrays covering WARMUP + (i1 - i0) bars.
    """
    pa = market.pairs[pair]
    lo = max(i0 - WARMUP, 1)
    src = np.arange(lo, i1)
    prev = pa.close[src - 1]
    shape = np.stack([pa.open[src] / prev, pa.high[src] / prev, pa.low[src] / prev, pa.close[src] / prev], axis=1)
    dts = market.dt_years[src]

    n = WARMUP + (i1 - i0)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(src) - BLOCK, size=n // BLOCK + 2)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
    rel = shape[idx]
    close = pa.close[lo - 1] * np.cumprod(rel[:, 3])
    prev_c = np.concatenate(([pa.close[lo - 1]], close[:-1]))
    o, h, l = prev_c * rel[:, 0], prev_c * rel[:, 1], prev_c * rel[:, 2]
    h = np.maximum.reduce([h, o, close])
    l = np.minimum.reduce([l, o, close])

    base, quote = PAIRS[pair]
    if quote == "USD":
        q = np.ones(n)
    elif base == "USD":
        q = 1.0 / close
    else:  # cross pair: keep the real conversion rate
        q = np.resize(pa.q_usd[src], n)
    vol = trailing_vol(pd.Series(close))
    return PairArrays(open=o, high=h, low=l, close=close, q_usd=q, vol=vol), dts[idx]


def synthetic_sharpes(market, pair, params, i0, i1, rates, eng, n_paths=N_SYNTHETIC, mode=1, s1_mode=None) -> np.ndarray:
    out = []
    for seed in range(n_paths):
        pa, dt = synthetic_pair(market, pair, i0, i1, seed)
        ret = nelstein_returns(pa, dt, WARMUP, len(dt), params, notional=eng["notional"],
                               commission_pct=eng["commission_pct"], slip_vol_mult=eng["slip_vol_mult"],
                               carry_diff_pct=carry_diff(pair, rates), mode=mode, s1_mode=s1_mode, cache=False)
        out.append(sharpe(ret))
    return np.array(out)


def state_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades and P&L by how they entered and whether they reached the trend state."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["Path"] = np.where(t["entry_state"] == "S2", "S2 entry (trend)",
                         np.where(t["reached_s2"], "S1 breakout that turned into S2", "S1 breakout that stayed S1"))
    rows = []
    for path, g in list(t.groupby("Path")) + [("All trades", t)]:
        s = trade_stats(g["net_pnl"].to_numpy(), g["hours_held"].to_numpy())
        rows.append({"Path": path, "Trades": s["trades"], "Win rate (%)": s["win_rate"] * 100,
                     "Net P&L ($)": g["net_pnl"].sum(), "Gross P&L ($)": (g["price_pnl"] + g["carry"]).sum(),
                     "Costs ($)": (g["slippage"] + g["commission"]).sum(), "Average hours held": s["avg_hours"]})
    return pd.DataFrame(rows)


def run_study(market: Market, frames, pairs, rates, tunes: dict, tune_rng, sim_rng, eng: dict) -> dict:
    """Everything the Nelstein1 study tab shows. tunes[pair] is the output of tuning.tune for that pair."""
    rates = dict(rates)
    t0, t1 = market.window(*tune_rng)
    s0, s1 = market.window(*sim_rng)
    grid = NELSTEIN.grid
    labels = [NELSTEIN.label(p) for p in grid]
    no_cost = {**eng, "commission_pct": 0.0, "slip_vol_mult": 0.0}
    double = {**eng, "commission_pct": eng["commission_pct"] * 2, "slip_vol_mult": eng["slip_vol_mult"] * 2}

    def run(pair, params, rng=(s0, s1), kw=eng, **extra):
        return backtest(market, frames, KEY, pair, params, rates, *rng, **kw, **extra)

    def ret(pair, params, kw=eng, **extra):
        return backtest_returns(market, frames, KEY, pair, params, rates, s0, s1, **kw, **extra)

    rows, equity, trades, grids, variants = [], {}, {}, {}, {}
    for pair in pairs:
        tn = tunes[pair]
        label = tn["suggested"]
        params = grid[labels.index(label)]

        insample = run(pair, params, rng=(t0, t1))
        hold = run(pair, params)
        hourly = hold.hourly
        equity[pair] = hourly["equity"]
        trades[pair] = hold.trades
        v = {
            "net": hourly["ret"].to_numpy(),
            "gross": ret(pair, params, kw=no_cost),
            "double": ret(pair, params, kw=double),
            "delay": ret(pair, params, delay=1),
            "reverse": ret(pair, params, mode=-1),
            "fade_s1": ret(pair, params, s1_mode=-1),
            "k_low": ret(pair, params, trend_k=TREND_K - 0.5),
            "k_high": ret(pair, params, trend_k=TREND_K + 0.5),
        }
        variants[pair] = v

        # Overfitting: does a good tuning-period score predict a good holdout score?
        tune_score = tn["params"].set_index("Parameters")["Full-period score"].reindex(labels).to_numpy()
        hold_sharpe = np.array([sharpe(ret(pair, p)) for p in grid])
        ok = np.isfinite(tune_score) & np.isfinite(hold_sharpe)
        rho = spearmanr(tune_score[ok], hold_sharpe[ok]).statistic if ok.sum() > 3 else float("nan")
        chosen_hold = hold_sharpe[labels.index(label)]
        grids[pair] = pd.DataFrame({"Parameters": labels, "x": [p[0] for p in grid], "y": [p[1] for p in grid],
                                    "w": [p[2] for p in grid], "z": [p[3] for p in grid],
                                    "Tuning score": tune_score, "Holdout Sharpe": hold_sharpe})

        synth = synthetic_sharpes(market, pair, params, s0, s1, rates, eng)
        oos = tn["oos"]
        wf = tn["wf"]
        hm = _metrics(hold)
        tr = hold.trades
        gross_wins = (tr["net_pnl"].clip(lower=0)).sum()
        top5 = tr["net_pnl"].nlargest(5).clip(lower=0).sum() / gross_wins if gross_wins > 0 else float("nan")
        rows.append({
            "Pair": pair, "Parameters": label, "x": params[0], "y": params[1], "w": params[2], "z": params[3],
            "In-sample Sharpe": _metrics(insample)["sharpe"],
            "Walk-forward OOS Sharpe": sharpe(oos.to_numpy()) if len(oos) else float("nan"),
            "WF windows positive (%)": float((wf["Out-of-sample"] > 0).mean() * 100) if len(wf) else float("nan"),
            "WF parameter sets used": int(wf["Chosen parameters"].nunique()) if len(wf) else 0,
            "Holdout Sharpe": hm["sharpe"], "Holdout return (%)": hm["total_return"] * 100,
            "Holdout max drawdown (%)": hm["max_drawdown"] * 100, "Trades": hm["trades"],
            "Win rate (%)": hm["win_rate"] * 100, "Profit factor": hm["profit_factor"],
            "Gross Sharpe": sharpe(v["gross"]), "Sharpe, costs doubled": sharpe(v["double"]),
            "Sharpe, one bar late": sharpe(v["delay"]), "Sharpe, reversed": sharpe(v["reverse"]),
            "Sharpe, S1 breakouts faded": sharpe(v["fade_s1"]),
            "Sharpe, trend cut-off low": sharpe(v["k_low"]), "Sharpe, trend cut-off high": sharpe(v["k_high"]),
            "Random-walk median Sharpe": float(np.median(synth)),
            "Random-walk p-value": float((np.sum(synth >= hm["sharpe"]) + 1) / (len(synth) + 1)),
            "Rank correlation tuning vs holdout": rho,
            "Chosen setting's holdout rank (%)": float((hold_sharpe[ok] < chosen_hold).mean() * 100),
            "Grid median holdout Sharpe": float(np.nanmedian(hold_sharpe)),
            "Top 5 trades' share of profits (%)": top5 * 100,
            "Long P&L ($)": tr.loc[tr["direction"] > 0, "net_pnl"].sum(),
            "Short P&L ($)": tr.loc[tr["direction"] < 0, "net_pnl"].sum(),
        })

    table = pd.DataFrame(rows)
    combined = {name: sharpe(np.mean([variants[p][name] for p in pairs], axis=0))
                for name in ("net", "gross", "double", "delay", "reverse", "fade_s1", "k_low", "k_high")}
    all_trades = pd.concat([trades[p] for p in pairs], ignore_index=True) if pairs else pd.DataFrame()
    premises = pd.concat([premise_test(market, p, rows[k]["x"], rows[k]["y"], s1) for k, p in enumerate(pairs)],
                         ignore_index=True) if pairs else pd.DataFrame()
    return {"table": table, "combined": combined, "equity": equity, "trades": trades,
            "states": state_breakdown(all_trades), "grids": grids, "premises": premises}


def premise_test(market: Market, pair: str, x: int, y: float, i_end: int, horizon: int = 24) -> pd.DataFrame:
    """Do the two signals predict the next day's move, before any stop or cost?

    For every bar flagged as trending, and every range bar that closes beyond the band, the move
    from the next open to the close `horizon` bars later is signed by the signal's direction.
    A positive average means the signal points the right way on average. Samples overlap, so
    the t-statistic uses an effective count of n / horizon.
    """
    pa = market.pairs[pair]
    c = pa.close
    trend, bmean, bmad, _, _ = features_cached(c, int(x))
    n = min(i_end, len(c))
    t = np.arange(x + 1, n - horizon - 1)
    fwd = c[t + horizon] / pa.open[t + 1] - 1.0
    breakout = np.where(c[t] > bmean[t] + y * bmad[t], 1, np.where(c[t] < bmean[t] - y * bmad[t], -1, 0))
    breakout = np.where(trend[t] == 0, breakout, 0)
    rows = []
    for name, sig in (("S2 trend signal (follow it)", trend[t].astype(int)), ("S1 band breakout (follow it)", breakout)):
        on = sig != 0
        signed = sig[on] * fwd[on]
        k = int(on.sum())
        eff = max(k / horizon, 1.0)
        mean = float(signed.mean()) if k else float("nan")
        sd = float(signed.std(ddof=1)) if k > 1 else float("nan")
        rows.append({"Pair": pair, "Signal": name, "Bars flagged": k,
                     "Average next-day move in signal direction (bp)": mean * 1e4,
                     "Right direction (%)": float((signed > 0).mean() * 100) if k else float("nan"),
                     "t-statistic (approx.)": mean / (sd / np.sqrt(eff)) if k > 1 and sd > 0 else float("nan")})
    return pd.DataFrame(rows)


def _row_metrics(market, frames, pair, params, rates, rngs, eng, mode, s1_mode) -> tuple[dict, pd.Series, pd.DataFrame]:
    (t0, t1), (s0, s1) = rngs
    kw = dict(mode=mode, s1_mode=s1_mode)
    ins = backtest(market, frames, KEY, pair, params, rates, t0, t1, **eng, **kw)
    hold = backtest(market, frames, KEY, pair, params, rates, s0, s1, **eng, **kw)
    no_cost = {**eng, "commission_pct": 0.0, "slip_vol_mult": 0.0}
    gross = backtest_returns(market, frames, KEY, pair, params, rates, s0, s1, **no_cost, **kw)
    hm = _metrics(hold)
    return ({
        "Sharpe, tuning period": _metrics(ins)["sharpe"], "Holdout Sharpe": hm["sharpe"],
        "Holdout return (%)": hm["total_return"] * 100, "Holdout max drawdown (%)": hm["max_drawdown"] * 100,
        "Trades": hm["trades"], "Win rate (%)": hm["win_rate"] * 100, "Profit factor": hm["profit_factor"],
        "Gross Sharpe": sharpe(gross),
    }, hold.hourly["equity"], hold.hourly)


def custom_runs(market: Market, frames, rows, rates, tune_rng, sim_rng, eng: dict) -> dict:
    """Backtest hand-picked settings. rows: (pair, x, y, w, z, direction) per pair."""
    rates = dict(rates)
    rngs = (market.window(*tune_rng), market.window(*sim_rng))
    table, equity, rets = [], {}, []
    for pair, x, y, w, z, direction in rows:
        params = (int(x), float(y), float(w), float(z))
        mode, s1_mode = DIRECTIONS[direction]
        m, eq, hourly = _row_metrics(market, frames, pair, params, rates, rngs, eng, mode, s1_mode)
        table.append({"Pair": pair, "Settings": NELSTEIN.label(params), "Direction": direction, **m})
        equity[pair] = eq
        rets.append(hourly["ret"].to_numpy())
    combined = sharpe(np.mean(rets, axis=0)) if rets else float("nan")
    return {"table": pd.DataFrame(table), "equity": equity, "combined": combined}


def reversal_test(market: Market, frames, pair, rates, base_tune: dict, tune_rng, sim_rng, eng: dict,
                  tune_kwargs: dict) -> dict:
    """The strategy as specified against reversed versions on one pair.

    The fair rows re-tune the reversed rules on the tuning period and only then look at the holdout.
    The hindsight row flips the already-tuned settings because the holdout lost money, which uses
    the holdout to make the choice, so its holdout result is not out-of-sample.
    """
    rates = dict(rates)
    t0, t1 = market.window(*tune_rng)
    s0, s1 = market.window(*sim_rng)
    rngs = ((t0, t1), (s0, s1))
    labels = [NELSTEIN.label(p) for p in NELSTEIN.grid]
    base_params = NELSTEIN.grid[labels.index(base_tune["suggested"])]

    variants = [
        ("As specified, tuned", "As specified", base_tune, base_params, "fair"),
        ("Reversed, same settings", "Reversed", None, base_params, "hindsight"),
    ]
    for name, direction in (("Reversed, re-tuned", "Reversed"), ("S1 faded, re-tuned", "S1 faded")):
        mode, s1_mode = DIRECTIONS[direction]
        tn = tune(market, frames, KEY, [pair], rates, t0, t1, mode=mode, s1_mode=s1_mode, **tune_kwargs, **eng)
        variants.append((name, direction, tn, NELSTEIN.grid[labels.index(tn["suggested"])], "fair"))

    rows, equity = [], {}
    for name, direction, tn, params, kind in variants:
        mode, s1_mode = DIRECTIONS[direction]
        m, eq, _ = _row_metrics(market, frames, pair, params, rates, rngs, eng, mode, s1_mode)
        synth = synthetic_sharpes(market, pair, params, s0, s1, rates, eng, mode=mode, s1_mode=s1_mode)
        oos = tn["oos"] if tn is not None else None
        rows.append({
            "Version": name, "Settings": NELSTEIN.label(params),
            "Holdout is out-of-sample": "yes" if kind == "fair" else "no (chosen after seeing it)",
            "Sharpe, tuning period": m["Sharpe, tuning period"],
            "Walk-forward OOS Sharpe": sharpe(oos.to_numpy()) if oos is not None and len(oos) else float("nan"),
            **{k: v for k, v in m.items() if k != "Sharpe, tuning period"},
            "Random-walk p-value": float((np.sum(synth >= m["Holdout Sharpe"]) + 1) / (len(synth) + 1)),
        })
        equity[name] = eq
    return {"table": pd.DataFrame(rows), "equity": equity}


# y x w search with x and z held fixed
Y_GRID = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0)
W_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
SEARCH_OBJECTIVES = {"total_return": "Total return", "sharpe": "Sharpe ratio", "calmar": "Return ÷ max drawdown (Calmar)"}


def yw_search(market: Market, frames, setups, rates, tune_rng, sim_rng, eng: dict, objective: str) -> dict:
    """For each pair, hold x and z fixed and try every (y, w) on the grid in both periods.

    setups: (pair, x, z, direction) per pair. The pick is made on the tuning period only; the
    holdout shows how that pick did on later data. The holdout's own best is reported too, but
    choosing it uses the holdout, so it is hindsight.
    """
    rates = dict(rates)
    t0, t1 = market.window(*tune_rng)
    s0, s1 = market.window(*sim_rng)
    grids, summary, equity = {}, [], {}
    for pair, x, z, direction in setups:
        rows = []
        for y in Y_GRID:
            for w in W_GRID:
                params = (int(x), y, w, float(z), direction)
                mt = compute_measures(backtest_returns(market, frames, KEY, pair, params, rates, t0, t1, **eng))
                mh = compute_measures(backtest_returns(market, frames, KEY, pair, params, rates, s0, s1, **eng))
                rows.append({"y": y, "w": w,
                             "tune_total_return": mt["total_return"], "tune_sharpe": mt["sharpe"],
                             "tune_calmar": mt["calmar"], "hold_total_return": mh["total_return"],
                             "hold_sharpe": mh["sharpe"], "hold_calmar": mh["calmar"],
                             "hold_max_drawdown": mh["max_drawdown"]})
        g = pd.DataFrame(rows)
        grids[pair] = g
        tune_col, hold_col = f"tune_{objective}", f"hold_{objective}"
        pick = g.loc[g[tune_col].fillna(-np.inf).idxmax()]
        hind = g.loc[g[hold_col].fillna(-np.inf).idxmax()]
        params = (int(x), float(pick["y"]), float(pick["w"]), float(z), direction)
        res = backtest(market, frames, KEY, pair, params, rates, s0, s1, **eng)
        equity[pair] = res.hourly["equity"]
        positive = float((g[hold_col] > 0).mean() * 100)
        summary.append({
            "Pair": pair, "x (fixed)": int(x), "z (fixed)": float(z), "Direction": direction,
            "Best y": float(pick["y"]), "Best w": float(pick["w"]),
            "Tuning-period return (%)": pick["tune_total_return"] * 100, "Tuning-period Sharpe": pick["tune_sharpe"],
            "Holdout return (%)": pick["hold_total_return"] * 100, "Holdout Sharpe": pick["hold_sharpe"],
            "Holdout max drawdown (%)": pick["hold_max_drawdown"] * 100, "Holdout trades": int(len(res.trades)),
            "Holdout-best y": float(hind["y"]), "Holdout-best w": float(hind["w"]),
            "Holdout-best return (%)": hind["hold_total_return"] * 100,
            "Settings positive on holdout (%)": positive,
            "Rank correlation tuning vs holdout": spearmanr(g[tune_col], g[hold_col], nan_policy="omit").statistic,
        })
    return {"summary": pd.DataFrame(summary), "grids": grids, "equity": equity, "objective": objective}
