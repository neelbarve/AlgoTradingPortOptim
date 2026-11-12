"""Strategy tuning: grid search with walk-forward optimization and K-fold cross-validation.

Each parameter set is backtested once over the whole tuning period. Its hourly
returns (pooled across the tuning pairs at equal notional) are then sliced into
training and test windows. Slicing lets a trade span a window boundary, so the
K-fold scheme drops an embargo of bars on each side of the test block.
"""
import numpy as np
import pandas as pd

from .data import Market
from .engine import run_backtest
from .measures import compute_measures, utility
from .strategies import STRATEGIES, carry_diff, make_signal


def returns_matrix(
    market: Market,
    frames: dict[str, pd.DataFrame],
    strategy_key: str,
    pairs: list[str],
    rates: dict[str, float],
    i0: int,
    i1: int,
    **engine_kwargs,
) -> pd.DataFrame:
    """Hourly pooled returns: one column per parameter set in the strategy's grid."""
    strat = STRATEGIES[strategy_key]
    out = {}
    for params in strat.grid:
        pooled = np.zeros(i1 - i0)
        for pair in pairs:
            sig = make_signal(strategy_key, frames[pair], pair, params, rates)
            res = run_backtest(
                market, pair, sig, i0, i1,
                carry_diff_pct=carry_diff(pair, rates), **engine_kwargs,
            )
            pooled += res.hourly["ret"].to_numpy()
        out[strat.label(params)] = pooled / len(pairs)
    return pd.DataFrame(out, index=market.index[i0:i1])


def _score(r: np.ndarray, measure: str, cash_rate: float) -> float:
    return compute_measures(r, cash_rate=cash_rate)[measure]


def _best(R: pd.DataFrame, mask: np.ndarray, measure: str, cash_rate: float) -> tuple[str, float]:
    """Parameter set with the best score on the masked rows."""
    best_label, best_value, best_u = R.columns[0], float("nan"), -np.inf
    for label in R.columns:
        v = _score(R[label].to_numpy()[mask], measure, cash_rate)
        u = utility(v, measure)
        if u > best_u:
            best_label, best_value, best_u = label, v, u
    return best_label, best_value


def walk_forward(
    R: pd.DataFrame, measure: str, train_months: int, test_months: int, cash_rate: float = 0.0
) -> tuple[pd.DataFrame, pd.Series]:
    """Tune on a trailing window, test on the next block, roll forward.

    Returns the per-window table and the stitched out-of-sample return series.
    """
    idx = R.index
    rows, oos_parts = [], []
    start = idx[0]
    while True:
        train_end = start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        train = np.asarray((idx >= start) & (idx < train_end))
        test = np.asarray((idx >= train_end) & (idx < test_end))
        if test.sum() < 24:
            break
        label, is_score = _best(R, train, measure, cash_rate)
        test_r = R[label][test]
        rows.append(
            {
                "Train from": start.date(), "Train to": (train_end - pd.Timedelta(days=1)).date(),
                "Test from": train_end.date(), "Test to": idx[test][-1].date(),
                "Chosen parameters": label,
                "In-sample": is_score,
                "Out-of-sample": _score(test_r.to_numpy(), measure, cash_rate),
            }
        )
        oos_parts.append(test_r)
        start = start + pd.DateOffset(months=test_months)
    oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    return pd.DataFrame(rows), oos


def kfold(
    R: pd.DataFrame, measure: str, k: int, embargo_bars: int, cash_rate: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K contiguous folds; each is the test block once.

    Training data is everything else, minus `embargo_bars` on each side of the
    test block. Returns the per-fold table and a per-parameter-set table of
    test-fold scores.
    """
    n = len(R)
    edges = np.linspace(0, n, k + 1).astype(int)
    pos = np.arange(n)
    fold_rows = []
    per_param = {label: [] for label in R.columns}
    for f in range(k):
        a, b = edges[f], edges[f + 1]
        test = (pos >= a) & (pos < b)
        train = (pos < a - embargo_bars) | (pos >= b + embargo_bars)
        label, is_score = _best(R, train, measure, cash_rate)
        for col in R.columns:
            per_param[col].append(_score(R[col].to_numpy()[test], measure, cash_rate))
        fold_rows.append(
            {
                "Fold": f + 1,
                "Test from": R.index[a].date(), "Test to": R.index[b - 1].date(),
                "Chosen parameters": label,
                "In-sample": is_score,
                "Out-of-sample": per_param[label][-1],
            }
        )
    param_rows = []
    for label, scores in per_param.items():
        arr = np.array(scores, dtype=float)
        ok = np.isfinite(arr)
        param_rows.append(
            {
                "Parameters": label,
                "Mean test-fold score": float(arr[ok].mean()) if ok.any() else float("nan"),
                "Worst fold": float(min(arr[ok], key=lambda v: utility(v, measure))) if ok.any() else float("nan"),
                "Full-period score": _score(R[label].to_numpy(), measure, cash_rate),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(param_rows)


def suggest_from_kfold(param_table: pd.DataFrame, measure: str) -> str:
    """Parameter set with the best mean test-fold score."""
    u = param_table["Mean test-fold score"].map(lambda v: utility(v, measure))
    return str(param_table.loc[u.idxmax(), "Parameters"])
