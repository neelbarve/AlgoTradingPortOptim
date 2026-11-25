"""The pipeline: data -> strategy blocks -> tuning -> estimation -> portfolio -> results.

Each stage is a plain function with a fixed signature, and each replaceable piece is a parameter:

  strategies  any block registered in fxsim.strategies (add a module there; see strategies/base.py)
  tuner       fxsim.tuning.tune by default
  estimator   fxsim.estimation.estimate_asset by default
  optimizer   fxsim.optimizer.optimize_weights by default
  schedule    fxsim.portfolio.clock_schedule by default (when the portfolio checks for rebalancing)

Swapping one piece never requires editing another. The Streamlit page calls these same stage
functions (adding caching around them); run_pipeline chains them for scripts and notebooks.
"""
from dataclasses import asdict, dataclass, field, replace

import pandas as pd

from .config import COMMISSION_PCT, REBALANCE_HOURS, SLIP_VOL_MULT, TRAIL_STOP_PCT
from .data import Market
from .estimation import estimate_asset
from .optimizer import optimize_weights
from .portfolio import Asset, clock_schedule, simulate
from .runner import backtest
from .strategies import STRATEGIES
from .tuning import tune

REBALANCE_INTERVALS = (0.5, 1, 2, 4, 6, 8, 12, 24, 30, 48)   # hours


@dataclass(frozen=True)
class Execution:
    notional: float = 100_000.0
    trail_pct: float = TRAIL_STOP_PCT
    target_pct: float = 0.0
    commission_pct: float = COMMISSION_PCT
    slip_vol_mult: float = SLIP_VOL_MULT

    def kwargs(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TuningSettings:
    measure: str = "sharpe"
    train_months: int = 12
    test_months: int = 3
    k: int = 5
    embargo: int = 48
    cash_rate: float = 0.04

    def kwargs(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioSettings:
    capital: float = 100_000.0
    leverage_cap: float = 2.0
    expo_cap: float = 2.0
    objective: str = "ratio"
    risk_cap: float = 0.01
    reward_floor: float = 0.0002
    min_trade_frac: float = 0.01
    cash_rate: float = 0.04
    rebalance_hours: float = REBALANCE_HOURS


@dataclass(frozen=True)
class PipelineConfig:
    strategies: tuple[str, ...]
    pairs: tuple[str, ...]
    rates: tuple[tuple[str, float], ...]
    tune_range: tuple
    est_range: tuple
    sim_range: tuple
    execution: Execution = field(default_factory=Execution)
    tuning: TuningSettings = field(default_factory=TuningSettings)
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    overrides: tuple = ()   # (key, pair, params): parameters that replace the tuned ones


# ---- Stage 1: tuning ------------------------------------------------------
def tune_stage(market: Market, frames, key: str, pairs, rates, tune_range, execution: Execution,
               tuning: TuningSettings, tuner=tune) -> dict:
    """Frozen parameters for one strategy, per pair or shared, depending on the block."""
    strat = STRATEGIES[key]
    i0, i1 = market.window(*tune_range)
    args = dict(**tuning.kwargs(), **execution.kwargs())
    if strat.per_pair:
        reports = {p: tuner(market, frames, key, [p], dict(rates), i0, i1, **args) for p in pairs}
        frozen = {p: strat.grid[r["labels"].index(r["suggested"])] for p, r in reports.items()}
    else:
        r = tuner(market, frames, key, list(pairs), dict(rates), i0, i1, **args)
        reports = {"*": r}
        shared = strat.grid[r["labels"].index(r["suggested"])]
        frozen = {p: shared for p in pairs}
    return {"frozen": frozen, "reports": reports}


# ---- Stage 2: estimation --------------------------------------------------
def estimate_stage(market: Market, frames, specs, rates, est_range, sim_range, execution: Execution,
                   cash_rate: float, leverage_cap: float, estimator=estimate_asset) -> tuple[list[Asset], pd.DataFrame]:
    """One portfolio asset per (strategy, pair, params): estimates from est_range, signals in sim_range."""
    e0, e1 = market.window(*est_range)
    s0, s1 = market.window(*sim_range)
    assets, rows = [], []
    for key, pair, params in specs:
        est_trades = backtest(market, frames, key, pair, params, rates, e0, e1, **execution.kwargs()).trades
        sim_trades = backtest(market, frames, key, pair, params, rates, s0, s1, **execution.kwargs()).trades
        est = estimator(est_trades, cash_rate, 1.0 / leverage_cap)
        name = f"{STRATEGIES[key].short} · {pair}"
        assets.append(Asset(name, key, pair, sim_trades, est["excess_ret"], est["std_ret"]))
        rows.append({"Asset": name, **est, "sim_trades": len(sim_trades)})
    return assets, pd.DataFrame(rows)


# ---- Stage 3: portfolio ---------------------------------------------------
def portfolio_stage(market: Market, assets, sim_range, rates, execution: Execution, settings: PortfolioSettings,
                    optimizer=optimize_weights, schedule=clock_schedule) -> dict:
    s0, s1 = market.window(*sim_range)
    return simulate(
        market, assets, s0, s1, capital=settings.capital, leverage_cap=settings.leverage_cap,
        expo_cap=settings.expo_cap, objective=settings.objective, risk_cap=settings.risk_cap,
        reward_floor=settings.reward_floor, commission_pct=execution.commission_pct,
        slip_vol_mult=execution.slip_vol_mult, min_trade_frac=settings.min_trade_frac,
        cash_rate=settings.cash_rate, rates=dict(rates), rebalance_hours=settings.rebalance_hours,
        optimizer=optimizer, schedule=schedule,
    )


def rebalance_sweep(market: Market, assets, sim_range, rates, execution: Execution, settings: PortfolioSettings,
                    intervals=REBALANCE_INTERVALS, optimizer=optimize_weights, schedule=clock_schedule) -> dict:
    """The same portfolio run once per rebalance-check interval (hours).

    Intervals that produce exactly the same check times (with hourly bars, anything up to an hour)
    share one run; their results carry same_as = the first interval with those checks.
    """
    s0, s1 = market.window(*sim_range)
    out, seen = {}, []
    for h in sorted(intervals, reverse=True):   # longest first, so a duplicate points at the coarser twin
        flags = (schedule or clock_schedule)(market.index, s0, s1, h)
        twin = next((h0 for h0, f0 in seen if (f0 == flags).all()), None)
        if twin is not None:
            out[h] = {**out[twin], "same_as": twin}
            continue
        seen.append((h, flags))
        out[h] = portfolio_stage(market, assets, sim_range, rates, execution,
                                 replace(settings, rebalance_hours=h), optimizer, schedule)
        out[h]["same_as"] = None
    return {h: out[h] for h in intervals}


# ---- The whole chain ------------------------------------------------------
def run_pipeline(market: Market, frames, cfg: PipelineConfig, *, tuner=tune, estimator=estimate_asset,
                 optimizer=optimize_weights, schedule=clock_schedule) -> dict:
    frozen, reports = {}, {}
    for key in cfg.strategies:
        t = tune_stage(market, frames, key, cfg.pairs, cfg.rates, cfg.tune_range, cfg.execution, cfg.tuning, tuner)
        frozen[key], reports[key] = t["frozen"], t["reports"]
    for key, pair, params in cfg.overrides:
        frozen[key][pair] = params
    specs = tuple((k, p, frozen[k][p]) for k in cfg.strategies for p in cfg.pairs)
    assets, estimates = estimate_stage(market, frames, specs, cfg.rates, cfg.est_range, cfg.sim_range,
                                       cfg.execution, cfg.portfolio.cash_rate, cfg.portfolio.leverage_cap, estimator)
    sim = portfolio_stage(market, assets, cfg.sim_range, cfg.rates, cfg.execution, cfg.portfolio, optimizer, schedule)
    return {"frozen": frozen, "tuning": reports, "specs": specs, "estimates": estimates, "simulation": sim}
