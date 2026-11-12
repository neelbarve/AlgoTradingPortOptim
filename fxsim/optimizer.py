"""Portfolio optimizer over the open signals at one rebalance.

Weights are USD notional / equity, never negative. Reward is the weighted
per-trade expected return; risk is the per-trade standard deviation of the
mix, treating trades as independent (per-trade measurement gives no
correlations). Co-movement is handled through currency exposure instead: the
net exposure to any non-USD currency is capped (decision D3).
"""
import numpy as np
from scipy.optimize import minimize

OBJECTIVES = {
    "ratio": "Best Sharpe ratio, held at the leverage cap",
    "max_reward": "Most reward with risk under a cap",
    "min_risk": "Least risk with reward above a floor",
}

_SCALE = 1e4  # work in basis points so the solver's tolerances bite


def portfolio_reward(w, mu):
    return float(np.dot(mu, w))


def portfolio_risk(w, sigma):
    return float(np.sqrt(np.sum((np.asarray(w) * sigma) ** 2)))


def _scale_to_limits(x, E, cap, expo_cap):
    """Largest multiple of mix x inside the leverage and currency-exposure caps."""
    k = cap / x.sum()
    if E.size:
        worst = np.abs(E @ x).max()
        if worst * k > expo_cap:
            k = expo_cap / worst
    return x * k


def _solve_max_reward(m, s, wc, E, cap, expo_cap, risk_cap, slip, x0):
    def cost(x):
        return float(np.sum(slip * np.sqrt((x - wc) ** 2 + 1e-14)))

    cons = [{"type": "ineq", "fun": lambda x: cap - x.sum()}]
    if E.size:
        cons.append({"type": "ineq", "fun": lambda x: expo_cap - E @ x})
        cons.append({"type": "ineq", "fun": lambda x: expo_cap + E @ x})
    if np.isfinite(risk_cap):
        cons.append({"type": "ineq", "fun": lambda x: 1.0 - np.sqrt(np.sum((x * s) ** 2) + 1e-18) / risk_cap})
    res = minimize(
        lambda x: -(m @ x - cost(x)) * _SCALE, x0,
        bounds=[(0.0, cap)] * len(m), constraints=cons, method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 300},
    )
    x = np.clip(res.x, 0.0, cap)
    ok = bool(res.success) and x.sum() <= cap * (1 + 1e-6)
    return (x if ok else x0), cost


def optimize_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_cur: np.ndarray,
    E: np.ndarray,
    *,
    cap: float,
    expo_cap: float,
    objective: str,
    risk_cap: float,
    reward_floor: float,
    slip: float,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Weights for the open signals.

    mu, sigma: per-trade expected (excess) return and standard deviation.
    w_cur: current weights. E: rows = non-USD currencies, columns = signals,
    entries +1 / -1 / 0 for the signal's exposure to that currency.
    slip: trading cost per unit of weight changed (a number, or one per signal).
    """
    n = len(mu)
    w = np.zeros(n)
    info = {"unreachable": False, "note": "", "reward": 0.0, "risk": 0.0}
    ok = (mu > 0) & np.isfinite(sigma) & (sigma > 0)
    if allowed is not None:
        ok &= allowed
    if not ok.any():
        info["note"] = "no open signal has a positive expected return"
        return w, info

    idx = np.where(ok)[0]
    m, s, wc, Ee = mu[idx], sigma[idx], w_cur[idx], E[:, idx]
    slip = np.broadcast_to(np.asarray(slip, dtype=float), (n,))[idx]  # cost per unit of weight traded
    mix = m / s**2  # best Sharpe mix when trades are independent
    at_cap = _scale_to_limits(mix, Ee, cap, expo_cap)

    if objective == "ratio":
        x = at_cap
    elif objective == "max_reward":
        x0 = at_cap.copy()
        r0 = portfolio_risk(x0, s)
        if r0 > risk_cap:
            x0 *= 0.95 * risk_cap / r0
        x, _ = _solve_max_reward(m, s, wc, Ee, cap, expo_cap, risk_cap, slip, x0)
    elif objective == "min_risk":
        # Highest reward reachable at all; if it misses the floor, flag it.
        x_top, cost = _solve_max_reward(m, s, wc, Ee, cap, expo_cap, np.inf, slip, at_cap)
        if m @ x_top - cost(x_top) < reward_floor:
            info["unreachable"] = True
            info["note"] = "reward floor is above anything the open signals can reach; positions kept"
            w = w_cur.copy()
            info["reward"], info["risk"] = portfolio_reward(w, mu), portfolio_risk(w[idx], s)
            return w, info
        cons = [
            {"type": "ineq", "fun": lambda x: cap - x.sum()},
            {"type": "ineq", "fun": lambda x: (m @ x - cost(x) - reward_floor) * _SCALE},
        ]
        if Ee.size:
            cons.append({"type": "ineq", "fun": lambda x: expo_cap - Ee @ x})
            cons.append({"type": "ineq", "fun": lambda x: expo_cap + Ee @ x})
        res = minimize(
            lambda x: np.sqrt(np.sum((x * s) ** 2) + 1e-18) * _SCALE, x_top,
            bounds=[(0.0, cap)] * len(m), constraints=cons, method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 300},
        )
        x = np.clip(res.x, 0.0, cap) if res.success else x_top
    else:
        raise KeyError(objective)

    x = np.where(x < 1e-6, 0.0, x)
    w[idx] = x
    info["reward"], info["risk"] = portfolio_reward(x, m), portfolio_risk(x, s)
    return w, info


def frontier(mu, sigma, E, *, cap, expo_cap, points: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Efficient frontier (risk, reward) for one set of open signals, before costs."""
    ok = (mu > 0) & np.isfinite(sigma) & (sigma > 0)
    if not ok.any():
        return np.zeros(1), np.zeros(1)
    m, s, Ee = mu[ok], sigma[ok], E[:, ok]
    zero = np.zeros(len(m))
    at_cap = _scale_to_limits(m / s**2, Ee, cap, expo_cap)
    top, _ = _solve_max_reward(m, s, zero, Ee, cap, expo_cap, np.inf, 0.0, at_cap)
    r_max = portfolio_risk(top, s)
    risks, rewards = [0.0], [0.0]
    for target in np.linspace(0.0, r_max, points)[1:]:
        x0 = at_cap * min(1.0, 0.95 * target / max(portfolio_risk(at_cap, s), 1e-12))
        x, _ = _solve_max_reward(m, s, zero, Ee, cap, expo_cap, target, 0.0, x0)
        risks.append(portfolio_risk(x, s))
        rewards.append(portfolio_reward(x, m))
    return np.array(risks), np.array(rewards)
