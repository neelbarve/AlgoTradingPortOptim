"""Asset estimation: expected return and risk measured per trade (decision D2).

An asset is one strategy on one pair. Its return is the mean of its historical
per-trade returns and its risk is their standard deviation. Returns include
carry and are taken before slippage, because the portfolio optimizer charges
slippage itself.
"""
import numpy as np
import pandas as pd

HOURS_PER_CALENDAR_YEAR = 365.25 * 24


def estimate_asset(trades: pd.DataFrame, cash_rate: float = 0.0, margin_req: float = 1.0) -> dict:
    """Per-trade statistics for one asset.

    cash_rate is a yearly fraction. margin_req is the margin posted per dollar
    of notional: only that part of equity stops earning the cash rate while a
    trade is open, so it sets the hurdle a trade must clear.
    """
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "mean_ret": float("nan"), "std_ret": float("nan"),
            "mean_ret_net": float("nan"), "win_rate": float("nan"),
            "avg_hours": float("nan"), "excess_ret": float("nan"),
        }
    gross = trades["ret_gross"].to_numpy()
    avg_hours = float(trades["hours_held"].mean())
    mean = float(gross.mean())
    return {
        "n_trades": n,
        "mean_ret": mean,
        "std_ret": float(gross.std(ddof=1)) if n > 1 else float("nan"),
        "mean_ret_net": float(trades["ret_net"].mean()),
        "win_rate": float((trades["ret_net"] > 0).mean()),
        "avg_hours": avg_hours,
        # Return above the cash interest given up on the posted margin
        # over an average holding time.
        "excess_ret": mean - cash_rate * margin_req * avg_hours / HOURS_PER_CALENDAR_YEAR,
    }
