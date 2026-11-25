"""Seasonality in each traded pair, with models chosen by information criteria.

Two questions per pair, on hourly log returns (basis points):
  mean        Does the average return depend on the hour of day or day of week?
  volatility  Does the size of moves (log of |return| + 1 bp) depend on them?
Each is answered by fitting a set of candidate regressions (no seasonality, hour-of-day and
day-of-week dummies, daily and weekly Fourier cycles with several harmonics) and choosing the
one with the lowest BIC; AIC's choice is shown alongside. Models are fitted before SPLIT_DATE and
checked out of sample after it, and the strongest seasonal model is tested with
autocorrelation- and heteroskedasticity-robust (HAC) standard errors.

A third check fits seasonal ARIMA models to daily returns with a 5-day (weekly) season,
again chosen by BIC.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

SPLIT_DATE = "2024-01-01"
HAC_LAGS = 24
DAILY_K = (1, 2, 3, 4, 6, 8)
COMBO_K = ((2, 1), (2, 2), (4, 1), (4, 2), (8, 1), (8, 2))
SARIMA_ORDERS = ((0, 0), (1, 0), (0, 1), (1, 1))
SARIMA_SEASONAL = ((0, 0), (1, 0), (0, 1))
DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def hourly_returns(frame: pd.DataFrame) -> pd.Series:
    r = np.log(frame["close"]).diff() * 1e4
    return r.dropna()


def _fourier(values: np.ndarray, period: float, k: int, tag: str) -> pd.DataFrame:
    cols = {}
    for j in range(1, k + 1):
        cols[f"{tag}_sin{j}"] = np.sin(2 * np.pi * j * values / period)
        cols[f"{tag}_cos{j}"] = np.cos(2 * np.pi * j * values / period)
    return pd.DataFrame(cols)


def _candidates(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Seasonal regressors for every candidate model (the constant is added when fitting)."""
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    how = dow * 24 + hour                       # hour of the week
    hod = pd.get_dummies(pd.Categorical(hour, categories=range(24)), prefix="h", drop_first=True, dtype=float)
    days = sorted(set(dow))
    dwd = pd.get_dummies(pd.Categorical(dow, categories=days), prefix="d", drop_first=True, dtype=float)
    out = {"No seasonality": pd.DataFrame(index=range(len(index))),
           "Hour of day (dummies)": hod,
           "Day of week (dummies)": dwd,
           "Hour of day + day of week (dummies)": pd.concat([hod, dwd], axis=1)}
    for k in DAILY_K:
        out[f"Daily cycle, {k} harmonic{'s' if k > 1 else ''}"] = _fourier(hour, 24, k, "day")
    for k, kw in COMBO_K:
        out[f"Daily ({k}) + weekly ({kw}) cycles"] = pd.concat(
            [_fourier(hour, 24, k, "day"), _fourier(how, 168, kw, "week")], axis=1)
    return {name: X.reset_index(drop=True) for name, X in out.items()}


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def select_model(y: pd.Series, split: str = SPLIT_DATE) -> dict:
    """Fit every candidate before `split`, rank by BIC, and check the choice after `split`."""
    cands = _candidates(y.index)
    is_in = np.asarray(y.index < pd.Timestamp(split, tz=y.index.tz))
    yv = y.to_numpy()
    y_in, y_out = yv[is_in], yv[~is_in]
    n = len(y_in)
    base_mean = y_in.mean()
    sse_const_out = float(((y_out - base_mean) ** 2).sum()) if len(y_out) else float("nan")

    rows, fits = [], {}
    for name, X in cands.items():
        Xa = np.column_stack([np.ones(len(yv)), X.to_numpy()]) if X.shape[1] else np.ones((len(yv), 1))
        beta, rss = _ols(y_in, Xa[is_in])
        k = Xa.shape[1]
        ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
        pred_out = Xa[~is_in] @ beta
        r2_out = 1 - float(((y_out - pred_out) ** 2).sum()) / sse_const_out if len(y_out) else float("nan")
        rows.append({"Model": name, "Parameters": k, "Log-likelihood": ll,
                     "AIC": 2 * k - 2 * ll, "BIC": k * np.log(n) - 2 * ll, "Out-of-sample R² (%)": r2_out * 100})
        fits[name] = (Xa, beta)
    table = pd.DataFrame(rows)
    table["ΔBIC vs none"] = table["BIC"] - table.loc[table["Model"] == "No seasonality", "BIC"].iloc[0]
    table = table.sort_values("BIC").reset_index(drop=True)
    bic_pick = table.loc[0, "Model"]
    aic_pick = table.sort_values("AIC").iloc[0]["Model"]
    seasonal = table[table["Model"] != "No seasonality"].iloc[0]

    # HAC test of the strongest seasonal model: are all its seasonal terms zero?
    Xa, _ = fits[seasonal["Model"]]
    res = sm.OLS(y_in, Xa[is_in]).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    R = np.eye(Xa.shape[1])[1:]
    p_hac = float(res.f_test(R).pvalue)

    # Profile by hour of day (in sample): the data and the strongest seasonal model's fit.
    Xs, bs = fits[seasonal["Model"]]
    fitted = pd.Series(Xs[is_in] @ bs, index=y.index[is_in])
    yin = pd.Series(y_in, index=y.index[is_in])
    by_hour = pd.DataFrame({
        "mean": yin.groupby(yin.index.hour).mean(),
        "se": yin.groupby(yin.index.hour).std() / np.sqrt(yin.groupby(yin.index.hour).count()),
        "fitted": fitted.groupby(fitted.index.hour).mean(),
    })
    by_day = yin.groupby(yin.index.dayofweek).agg(["mean", "std", "count"])
    by_day["se"] = by_day["std"] / np.sqrt(by_day["count"])
    return {"table": table, "bic_pick": bic_pick, "aic_pick": aic_pick, "best_seasonal": seasonal["Model"],
            "best_seasonal_dbic": float(seasonal["ΔBIC vs none"]), "best_seasonal_r2_out": float(seasonal["Out-of-sample R² (%)"]),
            "hac_p": p_hac, "by_hour": by_hour, "by_day": by_day, "n_in": n, "n_out": int((~is_in).sum())}


def daily_returns(frame: pd.DataFrame) -> pd.Series:
    """Daily log returns (bp), with the trading day ending at 21:00 UTC (the New York close)."""
    close = frame["close"]
    day = (close.index + pd.Timedelta(hours=3)).normalize()
    daily = close.groupby(day).last()
    daily = daily[daily.index.dayofweek < 5]
    return (np.log(daily).diff() * 1e4).dropna()


def sarima_selection(r: pd.Series, split: str = SPLIT_DATE) -> dict:
    """Seasonal ARIMA (p,0,q)x(P,0,Q,5) on daily returns before `split`, chosen by BIC."""
    y = r[r.index < pd.Timestamp(split, tz=r.index.tz)].to_numpy()
    rows, fits = [], {}
    for p, q in SARIMA_ORDERS:
        for P, Q in SARIMA_SEASONAL:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = SARIMAX(y, order=(p, 0, q), seasonal_order=(P, 0, Q, 5), trend="c").fit(disp=False)
                except Exception:
                    continue
            name = f"({p},0,{q})×({P},0,{Q})5"
            rows.append({"Model": name, "Parameters": int(len(res.params)), "AIC": res.aic, "BIC": res.bic,
                         "Weekly seasonal terms": "yes" if P or Q else "no"})
            fits[name] = res
    table = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    pick = table.loc[0, "Model"]
    lb = acorr_ljungbox(fits[pick].resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    return {"table": table, "bic_pick": pick, "aic_pick": table.sort_values("AIC").iloc[0]["Model"],
            "seasonal": table.loc[0, "Weekly seasonal terms"] == "yes", "ljung_box_p": float(lb), "n": len(y)}


def analyse_pair(frame: pd.DataFrame) -> dict:
    r = hourly_returns(frame)
    vol = np.log(r.abs() + 1.0)
    return {"mean": select_model(r), "vol": select_model(vol), "sarima": sarima_selection(daily_returns(frame))}


def summary_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for pair, res in results.items():
        m, v, s = res["mean"], res["vol"], res["sarima"]
        rows.append({
            "Pair": pair,
            "Mean: BIC choice": m["bic_pick"], "Mean: AIC choice": m["aic_pick"],
            "Mean: best seasonal ΔBIC": m["best_seasonal_dbic"], "Mean: HAC p-value": m["hac_p"],
            "Mean: out-of-sample R² (%)": m["best_seasonal_r2_out"],
            "Volatility: BIC choice": v["bic_pick"], "Volatility: ΔBIC vs none": v["best_seasonal_dbic"],
            "Volatility: out-of-sample R² (%)": v["best_seasonal_r2_out"],
            "Daily SARIMA (BIC)": s["bic_pick"], "Weekly season in daily returns": "yes" if s["seasonal"] else "no",
            "Ljung-Box p (residuals)": s["ljung_box_p"],
        })
    return pd.DataFrame(rows)
