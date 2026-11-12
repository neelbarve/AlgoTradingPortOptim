"""Reward, risk and ratio measures on hourly returns."""
import numpy as np

from .config import HOURS_PER_YEAR

# key -> (label, kind, higher is better)
MEASURES = {
    "total_return": ("Total return", "Reward", True),
    "annual_return": ("Annualized return", "Reward", True),
    "std": ("Standard deviation", "Risk", False),
    "downside_dev": ("Downside deviation", "Risk", False),
    "max_drawdown": ("Max drawdown", "Risk", False),
    "cvar": ("CVaR (worst 5% of hours)", "Risk", False),
    "sharpe": ("Sharpe", "Ratio", True),
    "sortino": ("Sortino", "Ratio", True),
    "calmar": ("Calmar", "Ratio", True),
}

PERCENT_MEASURES = {"total_return", "annual_return", "std", "downside_dev", "max_drawdown", "cvar"}


def compute_measures(
    returns: np.ndarray,
    *,
    cash_rate: float = 0.0,
    compound: bool = False,
    cvar_level: float = 0.05,
    downside_threshold: float = 0.0,
    hours_per_year: int = HOURS_PER_YEAR,
) -> dict[str, float]:
    """All measures for a series of hourly returns.

    compound=False treats returns as P&L over a fixed capital (strategy
    backtests at fixed notional). compound=True chains them (portfolio equity).
    cash_rate is a yearly rate as a fraction, used as the risk-free rate.
    """
    r = np.asarray(returns, dtype=float)
    nan = float("nan")
    if r.size < 2:
        return {k: nan for k in MEASURES}

    equity = np.cumprod(1.0 + r) if compound else 1.0 + np.cumsum(r)
    total = float(equity[-1] - 1.0)
    years = r.size / hours_per_year
    annual = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 else -1.0

    std = float(r.std(ddof=1) * np.sqrt(hours_per_year))
    below = np.minimum(r - downside_threshold, 0.0)
    downside = float(np.sqrt(np.mean(below**2)) * np.sqrt(hours_per_year))

    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    max_dd = float(np.max(1.0 - equity / peak))

    k = max(int(np.floor(r.size * cvar_level)), 1)
    cvar = float(-np.sort(r)[:k].mean())

    def ratio(num: float, den: float) -> float:
        return num / den if den > 1e-12 else nan

    return {
        "total_return": total,
        "annual_return": float(annual),
        "std": std,
        "downside_dev": downside,
        "max_drawdown": max_dd,
        "cvar": cvar,
        "sharpe": ratio(annual - cash_rate, std),
        "sortino": ratio(annual - cash_rate, downside),
        "calmar": ratio(annual, max_dd),
    }


def drawdown_series(equity: np.ndarray, start: float) -> np.ndarray:
    """Fraction below the running peak at each bar (0 at a new high, negative below it)."""
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(np.concatenate(([start], equity)))[1:]
    return equity / peak - 1.0


def longest_underwater(drawdown: np.ndarray) -> int:
    """Longest run of consecutive bars spent below a previous peak."""
    longest = run = 0
    for below in np.asarray(drawdown) < -1e-12:
        run = run + 1 if below else 0
        longest = max(longest, run)
    return longest


def trade_stats(net_pnl: np.ndarray, hours: np.ndarray) -> dict[str, float]:
    """Summary of a list of closed trades from their net P&L (USD) and holding hours."""
    pnl = np.asarray(net_pnl, dtype=float)
    nan = float("nan")
    if pnl.size == 0:
        return {"trades": 0, "win_rate": nan, "profit_factor": nan, "avg_win": nan,
                "avg_loss": nan, "expectancy": nan, "best": nan, "worst": nan, "avg_hours": nan}
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_loss = -losses.sum()
    return {
        "trades": int(pnl.size),
        "win_rate": float(wins.size / pnl.size),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else nan,
        "avg_win": float(wins.mean()) if wins.size else nan,
        "avg_loss": float(losses.mean()) if losses.size else nan,
        "expectancy": float(pnl.mean()),
        "best": float(pnl.max()),
        "worst": float(pnl.min()),
        "avg_hours": float(np.mean(hours)),
    }


def utility(value: float, measure: str) -> float:
    """Turn a measure into 'higher is better'; missing values rank last."""
    if value is None or not np.isfinite(value):
        return -np.inf
    return value if MEASURES[measure][2] else -value
