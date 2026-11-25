"""Simulated FX Strategy Portfolio: trading simulation page.

Run with:  streamlit run app.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

from fxsim.config import (AFFILIATION, COMMISSION_PCT, FOUNDERS, HOURS_PER_YEAR, PAIRS,
                          PLACEHOLDER_RATES, REBALANCE_HOURS, SLIP_VOL_MULT, TRAIL_STOP_PCT,
                          VOL_WINDOW)
from fxsim.data import align, build_market, load_available, quality_table
from fxsim.runner import backtest
from fxsim.measures import (MEASURES, PERCENT_MEASURES, compute_measures, drawdown_series,
                            longest_underwater, trade_stats)
from fxsim.optimizer import OBJECTIVES
from fxsim.pipeline import Execution, PortfolioSettings, REBALANCE_INTERVALS, estimate_stage, portfolio_stage, rebalance_sweep
from fxsim.seasonality import DAY_NAMES, SPLIT_DATE, analyse_pair, summary_table
from fxsim.strategies import STRATEGIES, carry_diff
from fxsim.strategies.nelstein import TREND_K, features_cached
from fxsim.study import KEY as STUDY_KEY
from fxsim.study import (DIRECTIONS, N_SYNTHETIC, SEARCH_OBJECTIVES, W_GRID, Y_GRID, custom_runs,
                         reversal_test, run_study, yw_search)
from fxsim.tuning import tune

st.set_page_config(page_title="FX Strategy Portfolio", layout="wide")

BACKTEST_NOTIONAL = 100_000.0

# ----------------------------------------------------------------------------
# Chart styling. Colors follow the entity (pair or currency), never its rank.
# ----------------------------------------------------------------------------
try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False

_SLOTS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_SLOTS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
SLOTS = _SLOTS_DARK if DARK else _SLOTS_LIGHT
PAIR_COLOR = dict(zip(PAIRS, SLOTS))
CCY_COLOR = dict(zip(["EUR", "GBP", "AUD", "CAD", "JPY"], SLOTS))
INK = "#f0efec" if DARK else "#0b0b0b"
INK2 = "#c3c2b7" if DARK else "#52514e"
MUTED = "#898781"
GRID = "#2c2c2a" if DARK else "#e1e0d9"
AXIS = "#383835" if DARK else "#c3c2b7"
SURFACE = "#1a1a19" if DARK else "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style(fig: go.Figure, *, height: int = 340, ytitle: str = "", xtitle: str = "",
          ysuffix: str = "", yprefix: str = "", unified: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10),
        hovermode="x unified" if unified else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)),
        font=dict(family=FONT, size=13, color=INK2),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     title=dict(text=xtitle, font=dict(color=INK2)))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
                     ticksuffix=ysuffix, tickformat="$,.0f" if yprefix == "$" else None, title=dict(text=ytitle, font=dict(color=INK2)))
    return fig


_chart_count = [0]


def show(fig: go.Figure) -> None:
    # A running key keeps two identical charts (the equity curve appears on two tabs) distinct.
    _chart_count[0] += 1
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"chart_{_chart_count[0]}")


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def fmt_measure(key: str, value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%" if key in PERCENT_MEASURES else f"{value:.2f}"


def money(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def measure_tiles(m: dict, keys: list[str]) -> None:
    for col, key in zip(st.columns(len(keys)), keys):
        col.metric(MEASURES[key][0], fmt_measure(key, m[key]))


def trade_tiles(s: dict) -> None:
    c = st.columns(7)
    c[0].metric("Trades", f"{s['trades']:,}")
    c[1].metric("Win rate", "n/a" if not np.isfinite(s["win_rate"]) else f"{s['win_rate'] * 100:.0f}%")
    c[2].metric("Profit factor", "n/a" if not np.isfinite(s["profit_factor"]) else f"{s['profit_factor']:.2f}")
    c[3].metric("Average win", money(s["avg_win"]))
    c[4].metric("Average loss", money(s["avg_loss"]))
    c[5].metric("Net P&L per trade", money(s["expectancy"]))
    c[6].metric("Average hours held", "n/a" if not np.isfinite(s["avg_hours"]) else f"{s['avg_hours']:.0f}")


def equity_chart(index, equity, color: str, start: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=index, y=equity, mode="lines", line=dict(color=color, width=2),
                    hovertemplate="$%{y:,.0f}<extra></extra>")
    fig.add_hline(y=start, line=dict(color=MUTED, width=1, dash="dot"))
    return style(fig, ytitle="USD", yprefix="$")


def drawdown_chart(index, drawdown, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=index, y=np.asarray(drawdown) * 100, mode="lines", line=dict(color=color, width=2),
                    fill="tozeroy", fillcolor=rgba(color, 0.18),
                    hovertemplate="%{y:.2f}%<extra></extra>")
    return style(fig, height=220, ytitle="Below peak", ysuffix="%")


def drawdown_caption(index, drawdown) -> str:
    dd = np.asarray(drawdown)
    worst = int(np.argmin(dd))
    days = longest_underwater(dd) / 24.0
    return (f"Deepest drawdown {dd[worst] * 100:.2f}% on {index[worst]:%Y-%m-%d}. Longest stretch below a "
            f"previous peak: {days:,.0f} trading days. Now {dd[-1] * 100:.2f}% below peak.")


# ----------------------------------------------------------------------------
# Cached computation
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading candle files")
def get_market():
    raw, missing = load_available()
    frames = align(raw)
    return raw, frames, missing, build_market(frames)


def _engine_kwargs(eng: tuple) -> dict:
    trail, target, commission, slip_mult = eng
    return dict(notional=BACKTEST_NOTIONAL, trail_pct=trail, target_pct=target,
                commission_pct=commission, slip_vol_mult=slip_mult)


@st.cache_data(show_spinner="Tuning strategies")
def c_tune(key, pairs, rates, start, end, eng, measure, train_m, test_m, k, embargo, cash):
    _, frames, _, mk = get_market()
    i0, i1 = mk.window(start, end)
    return tune(mk, frames, key, list(pairs), dict(rates), i0, i1, measure=measure, train_months=train_m,
                test_months=test_m, k=k, embargo=embargo, cash_rate=cash, **_engine_kwargs(eng))


@st.cache_data(show_spinner="Running the Nelstein1 study")
def c_study(pairs, rates, tune_rng, sim_rng, eng, measure, train_m, test_m, k, embargo, cash):
    _, frames, _, mk = get_market()
    tunes = {p: c_tune(STUDY_KEY, (p,), rates, *tune_rng, eng, measure, train_m, test_m, k, embargo, cash)
             for p in pairs}
    return run_study(mk, frames, list(pairs), dict(rates), tunes, tune_rng, sim_rng, _engine_kwargs(eng))


@st.cache_data(show_spinner="Running your settings")
def c_custom(rows, rates, tune_rng, sim_rng, eng):
    _, frames, _, mk = get_market()
    return custom_runs(mk, frames, list(rows), dict(rates), tune_rng, sim_rng, _engine_kwargs(eng))


@st.cache_data(show_spinner="Searching y and w")
def c_yw(setups, rates, tune_rng, sim_rng, eng, objective):
    _, frames, _, mk = get_market()
    return yw_search(mk, frames, list(setups), dict(rates), tune_rng, sim_rng, _engine_kwargs(eng), objective)


@st.cache_data(show_spinner="Testing reversed versions")
def c_reversal(pair, rates, tune_rng, sim_rng, eng, measure, train_m, test_m, k, embargo, cash):
    _, frames, _, mk = get_market()
    base = c_tune(STUDY_KEY, (pair,), rates, *tune_rng, eng, measure, train_m, test_m, k, embargo, cash)
    tk = dict(measure=measure, train_months=train_m, test_months=test_m, k=k, embargo=embargo, cash_rate=cash)
    return reversal_test(mk, frames, pair, dict(rates), base, tune_rng, sim_rng, _engine_kwargs(eng), tk)


@st.cache_data(show_spinner=False)
def c_backtest(key, pair, params, rates, start, end, eng):
    _, frames, _, mk = get_market()
    i0, i1 = mk.window(start, end)
    res = backtest(mk, frames, key, pair, params, dict(rates), i0, i1, **_engine_kwargs(eng))
    return res.trades, res.hourly


@st.cache_data(show_spinner="Simulating the portfolio")
def c_simulate(specs, rates, est_range, sim_range, eng, capital, lev_cap, expo_cap, objective,
               risk_cap, reward_floor, min_trade, cash, rebalance_hours):
    _, frames, _, mk = get_market()
    execution = Execution(**_engine_kwargs(eng))
    assets, estimates = estimate_stage(mk, frames, specs, dict(rates), est_range, sim_range, execution, cash, lev_cap)
    settings = PortfolioSettings(capital, lev_cap, expo_cap, objective, risk_cap, reward_floor, min_trade, cash,
                                 rebalance_hours)
    out = portfolio_stage(mk, assets, sim_range, dict(rates), execution, settings)
    out["estimates"] = estimates
    return out


@st.cache_data(show_spinner="Running the portfolio at every rebalance interval")
def c_sweep(specs, rates, est_range, sim_range, eng, capital, lev_cap, expo_cap, objective,
            risk_cap, reward_floor, min_trade, cash):
    _, frames, _, mk = get_market()
    execution = Execution(**_engine_kwargs(eng))
    assets, _ = estimate_stage(mk, frames, specs, dict(rates), est_range, sim_range, execution, cash, lev_cap)
    settings = PortfolioSettings(capital, lev_cap, expo_cap, objective, risk_cap, reward_floor, min_trade, cash)
    runs = rebalance_sweep(mk, assets, sim_range, dict(rates), execution, settings, REBALANCE_INTERVALS)
    keep = ("hourly", "totals", "rebalances", "fills", "round_trips", "hours_over_cap", "same_as")
    return {h: {k: r[k] for k in keep} for h, r in runs.items()}


@st.cache_data(show_spinner="Fitting seasonal models (first run only, about a minute)", persist="disk")
def c_seasonality(pairs):
    _, frames, _, _ = get_market()
    return {p: analyse_pair(frames[p]) for p in pairs}


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:g} min"
    return f"{h:g} hour" + ("" if h == 1 else "s")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
raw, frames, missing, market = get_market()
available = list(frames)
first_day, last_day = market.index[0].date(), market.index[-1].date()


def date_range(label: str, start: dt.date, end: dt.date) -> tuple[dt.date, dt.date]:
    value = st.date_input(label, (start, end), min_value=first_day, max_value=last_day)
    if not isinstance(value, tuple) or len(value) != 2:
        st.info("Pick both a start and an end date.")
        st.stop()
    return value


with st.sidebar:
    about_box = st.container()  # filled below, once the settings it describes are known
    st.header("Settings")
    sel_strategies = st.multiselect("Strategies", list(STRATEGIES), default=list(STRATEGIES),
                                    format_func=lambda k: STRATEGIES[k].name)
    sel_pairs = st.multiselect("Pairs", available, default=available)

    with st.expander("Periods", expanded=True):
        tune_range = date_range("Tuning", dt.date(2020, 7, 1), dt.date(2022, 12, 31))
        est_range = date_range("Estimation", dt.date(2023, 1, 1), dt.date(2023, 12, 31))
        sim_range = date_range("Simulation", dt.date(2024, 1, 1), min(dt.date(2025, 6, 30), last_day))

    with st.expander("Execution and costs", expanded=True):
        commission_pct = st.number_input("Transaction cost per fill (% of notional)", 0.0, 1.0,
                                         COMMISSION_PCT, 0.005, format="%.3f")
        slip_mult = st.number_input(
            "Slippage (× hourly volatility)", 0.0, 2.0, SLIP_VOL_MULT, 0.01, format="%.2f",
            help=f"Slippage per fill = this multiple × the pair's standard deviation of hourly returns "
                 f"over the previous {VOL_WINDOW} bars. Volatile hours cost more to trade.")
        trail_pct = st.number_input("Trailing stop-loss (% behind best price, 0 = off)", 0.0, 20.0,
                                    TRAIL_STOP_PCT, 0.25)
        target_pct = st.number_input("Take-profit (% from entry, 0 = off)", 0.0, 20.0, 0.0, 0.25)

    with st.expander("Tuning"):
        tune_measure = st.selectbox("Objective", list(MEASURES), index=list(MEASURES).index("sharpe"),
                                    format_func=lambda k: f"{MEASURES[k][0]} ({MEASURES[k][1].lower()})")
        train_months = st.number_input("Walk-forward: training window (months)", 3, 36, 12)
        test_months = st.number_input("Walk-forward: test block (months)", 1, 12, 3)
        n_folds = st.number_input("K-fold: number of folds", 2, 10, 5)
        embargo = st.number_input("K-fold: embargo (bars each side of the test block)", 0, 500, 48)

    with st.expander("Portfolio"):
        capital = st.number_input("Starting capital (USD)", 1_000.0, 1e9, 100_000.0, 10_000.0)
        lev_cap = st.number_input("Leverage cap (total notional ÷ equity)", 0.5, 50.0, 2.0, 0.5)
        cash_pct = st.number_input("Cash rate on idle equity (% a year)", 0.0, 20.0, 4.0, 0.25)
        objective = st.selectbox("Objective", list(OBJECTIVES), format_func=lambda k: OBJECTIVES[k])
        reb_options = list(REBALANCE_INTERVALS)
        rebalance_hours = st.selectbox(
            "Check for rebalancing every", reb_options, index=reb_options.index(REBALANCE_HOURS),
            format_func=lambda h: fmt_hours(h),
            help="Checks fall on UTC clock hours divisible by this number. New signals are entered at the "
                 "first check after they appear. Exits and stop-losses never wait for a check.")
        risk_cap_pct = st.number_input("Risk cap (per-trade std, % of equity)", 0.05, 20.0, 1.0, 0.05,
                                       disabled=objective != "max_reward")
        floor_pct = st.number_input("Reward floor (per-trade return, % of equity)", 0.0, 5.0, 0.02, 0.01,
                                    format="%.3f", disabled=objective != "min_risk")
        expo_cap = st.number_input("Max net exposure per currency (× equity)", 0.1, 50.0, float(lev_cap), 0.5)
        min_trade_pct = st.number_input(
            "Smallest resize worth trading (% of equity)", 0.0, 20.0, 1.0, 0.5,
            help="A rebalance that would change an existing position by less than this is skipped. "
                 "Entries and full exits always trade.")

    with st.expander("Interest rates (placeholders)"):
        st.caption("Constant % a year per currency. Not from the dataset; replace with real series.")
        rates_in = {c: st.number_input(c, -5.0, 30.0, v, 0.05, key=f"rate_{c}")
                    for c, v in PLACEHOLDER_RATES.items()}

    # ---- About: what kind of strategy this is, and who is behind it.
    with about_box, st.expander("About"):
        st.write(
            "A multi-strategy currency portfolio. Trend-following and carry-trade signals run on the "
            "major FX pairs at the 1-hour timeframe. Every open trade is treated as a portfolio asset, "
            "and a leveraged margin account decides how much of each to hold."
        )
        pair_names = ", ".join("/".join(PAIRS[p]) for p in sel_pairs) or "none selected"
        signal_names = ", ".join(STRATEGIES[k].name for k in sel_strategies) or "none selected"
        stop_text = f"{trail_pct:g}% trailing stop on every trade" if trail_pct > 0 else "no stop-loss"
        own = [STRATEGIES[k].name for k in sel_strategies if STRATEGIES[k].own_stops]
        if own:
            stop_text += f" ({', '.join(own)} {'sets its' if len(own) == 1 else 'set their'} own stops)"
        st.markdown(
            f"- **Market:** {pair_names}\n"
            f"- **Signals:** {signal_names}\n"
            f"- **Sizing:** weights re-optimized at a rebalance check every {fmt_hours(rebalance_hours)}, "
            f"leverage capped at {lev_cap:g}×\n"
            f"- **Risk control:** {stop_text}; net exposure per currency capped at {expo_cap:g}× equity\n"
            f"- **Costs:** {commission_pct:g}% per fill plus slippage that rises with volatility\n"
            "- **Validation:** walk-forward and K-fold tuning, then a later out-of-sample simulation"
        )
        st.markdown("**Founders**")
        for person in FOUNDERS:
            st.markdown(person["name"])
            if person["about"]:
                st.caption(person["about"])
        st.caption(AFFILIATION)

if not sel_strategies or not sel_pairs:
    st.info("Select at least one strategy and one pair in the sidebar.")
    st.stop()

rates = tuple(sorted(rates_in.items()))
eng = (float(trail_pct), float(target_pct), float(commission_pct), float(slip_mult))
cash = cash_pct / 100.0
PERIODS = {"Tuning": tune_range, "Estimation": est_range, "Simulation": sim_range}

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("Simulated FX Strategy Portfolio")
st.caption(
    "Trading simulation on hourly currency data: strategies generate entries and exits, every fill pays "
    "costs, and a margin account rebalances across the open trades. All times are UTC, all results in USD."
)

ordered = sorted(PERIODS.items(), key=lambda kv: kv[1][0])
for (name_a, a), (name_b, b) in zip(ordered, ordered[1:]):
    if a[1] >= b[0]:
        st.warning(f"The {name_a.lower()} and {name_b.lower()} periods overlap, so results are not out-of-sample.")

(tab_over, tab_port, tab_reb, tab_method, tab_bt, tab_nel, tab_seas, tab_tune, tab_est, tab_ccy, tab_data,
 tab_notes) = st.tabs(
    ["Overview", "Portfolio", "Rebalancing", "Methodology", "Strategy trades", "Nelstein1 study", "Seasonality",
     "Tuning", "Estimation", "Currency risk", "Data", "Assumptions"]
)

def params_label(key: str, params: tuple) -> str:
    return STRATEGIES[key].label(params)


# ---- Tuning (computed first: everything downstream uses the frozen parameters)
frozen: dict[str, dict[str, tuple]] = {}   # strategy -> pair -> frozen parameters
with tab_tune:
    st.subheader("Strategy tuning")
    st.write(
        f"Each parameter set is backtested over the tuning period on the selected pairs "
        f"(equal notional, pooled), with all costs and the trailing stop applied. The objective is "
        f"**{MEASURES[tune_measure][0]}**. Two out-of-sample tools are offered: walk-forward "
        "optimization and K-fold cross-validation with an embargo around each test block."
    )
    for key in sel_strategies:
        strat = STRATEGIES[key]
        st.markdown(f"#### {strat.name}")
        st.caption(strat.description)

        if strat.per_pair:
            frozen[key] = {}
            rows = []
            wf_tables = {}
            # Settings run from the Nelstein1 study tab replace the tuned ones when the user asked for that.
            override = {}
            if key == STUDY_KEY and st.session_state.get("nel_apply_portfolio"):
                override = {r[0]: r for r in st.session_state.get("nel_applied") or ()}
            for pair in sel_pairs:
                t = c_tune(key, (pair,), rates, *tune_range, eng, tune_measure,
                           int(train_months), int(test_months), int(n_folds), int(embargo), cash)
                frozen[key][pair] = strat.grid[t["labels"].index(t["suggested"])]
                if pair in override:
                    _, ox, oy, ow, oz, od = override[pair]
                    frozen[key][pair] = (ox, oy, ow, oz) if od == "As specified" else (ox, oy, ow, oz, od)
                wf = t["wf"]
                wf_tables[pair] = wf
                kf = t["params"].set_index("Parameters")
                rows.append({
                    "Pair": pair, "Frozen parameters": t["suggested"],
                    "In use": params_label(key, frozen[key][pair]),
                    "K-fold mean test score": kf.loc[t["suggested"], "Mean test-fold score"],
                    "Tuning-period score": kf.loc[t["suggested"], "Full-period score"],
                    "Walk-forward windows above zero": (f"{int((wf['Out-of-sample'] > 0).sum())} of {len(wf)}"
                                                        if len(wf) else "n/a"),
                    "Parameter sets walk-forward chose": int(wf["Chosen parameters"].nunique()) if len(wf) else 0,
                })
            st.write(
                f"Tuned separately for each pair over {len(strat.grid)} settings: x bars of lookback, y × average "
                "reversion for the band, w × average pullback for the trend stop, z% for the breakout stop. "
                "The frozen set is the one with the best mean test-fold score in K-fold. The Nelstein1 study tab "
                "tests how these choices hold up."
            )
            if override:
                st.info("Nelstein1 is running your settings from the Nelstein1 study tab (the *In use* column), not "
                        "the tuned ones. Click **Reset to tuned** there to go back.", icon="ℹ️")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"K-fold mean test score": st.column_config.NumberColumn(format="%.2f"),
                                        "Tuning-period score": st.column_config.NumberColumn(format="%.2f")})
            with st.expander("Walk-forward windows, pair by pair"):
                for pair, wf in wf_tables.items():
                    st.markdown(f"**{pair}**")
                    st.dataframe(wf, hide_index=True, width="stretch")
            continue

        t = c_tune(key, tuple(sel_pairs), rates, *tune_range, eng, tune_measure,
                   int(train_months), int(test_months), int(n_folds), int(embargo), cash)
        choice = st.selectbox(
            "Frozen parameter set (default: best mean test-fold score in K-fold)",
            t["labels"], index=t["labels"].index(t["suggested"]), key=f"frozen_{key}_{t['suggested']}",
        )
        shared = strat.grid[t["labels"].index(choice)]
        frozen[key] = {pair: shared for pair in sel_pairs}

        left, right = st.columns(2)
        for col, title, table, xcol in (
            (left, "Walk-forward", t["wf"], "Test from"),
            (right, f"{int(n_folds)}-fold cross-validation", t["folds"], "Fold"),
        ):
            with col:
                st.markdown(f"**{title}**")
                if table.empty:
                    st.info("The tuning period is too short for these window lengths.")
                    continue
                x = table[xcol].astype(str)
                fig = go.Figure()
                fig.add_bar(x=x, y=table["In-sample"], name="In-sample", marker_color=SLOTS[0],
                            marker_line=dict(color=SURFACE, width=2))
                fig.add_bar(x=x, y=table["Out-of-sample"], name="Out-of-sample", marker_color=SLOTS[1],
                            marker_line=dict(color=SURFACE, width=2))
                fig.update_layout(barmode="group", bargap=0.35)
                show(style(fig, height=280, ytitle=MEASURES[tune_measure][0],
                           xtitle="Test block starting" if xcol == "Test from" else "Test fold"))
                st.dataframe(table, hide_index=True, width="stretch")
        if not t["wf"].empty:
            oos = t["wf"]["Out-of-sample"]
            st.caption(
                f"Walk-forward: the out-of-sample score was above zero in "
                f"{int((oos > 0).sum())} of {len(oos)} test blocks. Latest window chose "
                f"{t['wf']['Chosen parameters'].iloc[-1]}."
            )
        with st.expander("Every parameter set tried (K-fold scores)"):
            st.dataframe(t["params"], hide_index=True, width="stretch")

# ---- Estimation and simulation
specs = tuple((key, pair, frozen[key][pair]) for key in sel_strategies for pair in sel_pairs)


def params_text(key: str) -> str:
    """The frozen parameters of a strategy, as one line."""
    sets = frozen[key]
    if len(set(sets.values())) == 1:
        return params_label(key, next(iter(sets.values())))
    return "; ".join(f"{pair} {params_label(key, v)}" for pair, v in sets.items())

sim = c_simulate(specs, rates, est_range, sim_range, eng, float(capital), float(lev_cap), float(expo_cap),
                 objective, risk_cap_pct / 100.0, floor_pct / 100.0, min_trade_pct / 100.0, cash,
                 float(rebalance_hours))
est = sim["estimates"]
hourly = sim["hourly"]
tot = sim["totals"]
trips = sim["round_trips"]
port_m = compute_measures(hourly["ret"].to_numpy(), cash_rate=cash, compound=True)
port_stats = trade_stats(trips["net_pnl"].to_numpy(), trips["hours_held"].to_numpy())
final_equity = float(hourly["equity"].iloc[-1])

# ---- Overview
with tab_over:
    st.subheader("Simulation result")
    st.write(f"{sim_range[0]} to {sim_range[1]}, starting with {capital:,.0f} USD, "
             f"leverage capped at {lev_cap:g}×.")
    c = st.columns(4)
    c[0].metric("Final equity", money(final_equity))
    c[1].metric("Total return", fmt_measure("total_return", port_m["total_return"]))
    c[2].metric("Max drawdown", fmt_measure("max_drawdown", port_m["max_drawdown"]))
    c[3].metric("Sharpe", fmt_measure("sharpe", port_m["sharpe"]))
    c = st.columns(4)
    c[0].metric("Trades taken", f"{port_stats['trades']:,}")
    c[1].metric("Win rate", "n/a" if not port_stats["trades"] else f"{port_stats['win_rate'] * 100:.0f}%")
    c[2].metric("Fills sent to market", f"{len(sim['fills']):,}")
    c[3].metric("Costs paid", money(tot["slippage"] + tot["commission"]))
    st.markdown("**Portfolio equity**")
    show(equity_chart(hourly.index, hourly["equity"], SLOTS[0], capital))
    if missing:
        st.info(f"Not loaded yet: {', '.join(missing)}. Add a file named like "
                f"`{missing[0]}_1h_….csv` to `currencydata_raw` and it joins automatically.")
    stages = [
        ("Portfolio", "The account: equity curve, drawdown, positions, every trade and fill, rebalance log."),
        ("Rebalancing", "The portfolio run with checks every 30 minutes to every 48 hours, compared."),
        ("Methodology", "Block diagram and every rule, from the data to entries, stop-losses and rebalancing."),
        ("Nelstein1 study", "Does the Nelstein1 strategy work? Holdout results, bias checks, benchmarks, overfitting."),
        ("Seasonality", "Hour-of-day, day-of-week and weekly patterns in each pair, models chosen by AIC/BIC."),
        ("Strategy trades", "Each strategy on each pair: equity curve, entries and exits on the price chart, trade list."),
        ("Tuning", "Parameter search scored by walk-forward optimization and K-fold cross-validation."),
        ("Estimation", "Each strategy–pair's expected return and risk, measured per trade."),
        ("Currency risk", "Net exposure per currency over time, and conversion of profits to USD."),
        ("Data", "The hourly candle files, their checks, and what a fill costs on each pair."),
    ]
    st.table(pd.DataFrame(stages, columns=["Tab", "What it shows"]).set_index("Tab"))

# ---- Portfolio
with tab_port:
    st.subheader("Portfolio simulation")
    st.write(
        f"Objective: **{OBJECTIVES[objective]}**. The account checks for rebalancing every "
        f"{fmt_hours(rebalance_hours)} and re-optimizes over the open signals "
        "when they have changed. Exits and stop-losses close at once. It holds one net position per pair. "
        "The Methodology tab has the full rules."
    )
    measure_tiles(port_m, ["total_return", "annual_return", "std", "max_drawdown", "sharpe", "sortino", "calmar"])
    c = st.columns(6)
    c[0].metric("Final equity", money(final_equity))
    c[1].metric("Price P&L", money(tot["price_pnl"]))
    c[2].metric("Carry", money(tot["carry"]))
    c[3].metric("Interest on idle equity", money(tot["interest"]))
    c[4].metric("Slippage paid", money(tot["slippage"]))
    c[5].metric("Transaction costs paid", money(tot["commission"]))

    st.markdown("**Equity curve**")
    show(equity_chart(hourly.index, hourly["equity"], SLOTS[0], capital))
    st.markdown("**Drawdown**")
    show(drawdown_chart(hourly.index, hourly["drawdown"], SLOTS[0]))
    st.caption(drawdown_caption(hourly.index, hourly["drawdown"]))

    pnl_pairs = sim["pair_pnl"].resample("1D").last().dropna(how="all")
    fig = go.Figure()
    for p in pnl_pairs.columns:
        fig.add_scatter(x=pnl_pairs.index, y=pnl_pairs[p], name=p, mode="lines",
                        line=dict(color=PAIR_COLOR[p], width=2),
                        hovertemplate="$%{y:,.0f}<extra>" + p + "</extra>")
    st.markdown("**Equity curve by pair: cumulative net P&L after costs**")
    show(style(fig, ytitle="USD", yprefix="$"))

    positions = sim["positions"]
    fig = go.Figure()
    for p in positions.columns:
        fig.add_scatter(x=positions.index, y=positions[p], name=p, mode="lines",
                        line=dict(color=PAIR_COLOR[p], width=2, shape="hv"),
                        hovertemplate="%{y:+.2f}×<extra>" + p + "</extra>")
    st.markdown("**Position in each pair (× equity; positive = long, negative = short)**")
    show(style(fig, height=300, ysuffix="×"))
    over = sim["hours_over_cap"]
    st.caption(
        f"Each step is a rebalance. Total leverage averaged {hourly['leverage'].mean():.2f}× and peaked at "
        f"{hourly['leverage'].max():.2f}× against a cap of {lev_cap:g}×"
        + (f"; it sat above the cap for {over:,} of {len(hourly):,} hours because the cap is enforced only "
           "at rebalances." if over else ".")
    )

    st.markdown("#### Trades")
    st.write("One row per position the account actually held, from the fill that opened it to the fill "
             "that closed it. P&L is after slippage and transaction costs.")
    if trips.empty:
        st.info("The account never opened a position: no open signal had a positive expected return.")
    else:
        trade_tiles(port_stats)
        st.dataframe(
            trips, hide_index=True, width="stretch", height=320,
            column_config={
                "asset": "Strategy · pair", "pair": None, "side": "Side",
                "entry_time": st.column_config.DatetimeColumn("Entry", format="YYYY-MM-DD HH:mm"),
                "exit_time": st.column_config.DatetimeColumn("Exit", format="YYYY-MM-DD HH:mm"),
                "hours_held": st.column_config.NumberColumn("Hours held", format="%.0f"),
                "entry_price": "Entry price", "exit_price": "Exit price", "exit_reason": "Exit reason",
                "peak_notional": st.column_config.NumberColumn("Largest size ($)", format="%.0f"),
                "price_pnl": st.column_config.NumberColumn("Price P&L ($)", format="%.0f"),
                "carry": st.column_config.NumberColumn("Carry ($)", format="%.1f"),
                "costs": st.column_config.NumberColumn("Costs ($)", format="%.1f"),
                "net_pnl": st.column_config.NumberColumn("Net P&L ($)", format="%.0f"),
            },
        )
        reasons = trips["exit_reason"].value_counts()
        st.caption("Exits: " + ", ".join(f"{n:,} by {r}" for r, n in reasons.items()) + ".")

    reb = sim["rebalances"]
    n_unreach = int(reb["unreachable"].sum()) if not reb.empty else 0
    st.markdown(f"#### Rebalance log ({len(reb):,} rebalances, {n_unreach:,} flagged unreachable)")
    st.dataframe(
        reb, hide_index=True, width="stretch", height=300,
        column_config={
            "time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
            "trigger": "Trigger", "open_signals": "Open signals",
            "old_weights": "Old weights (× equity)", "new_weights": "New weights (× equity)",
            "reward": st.column_config.NumberColumn("Expected return", format="%.5f"),
            "risk": st.column_config.NumberColumn("Risk", format="%.5f"),
            "unreachable": "Unreachable", "note": "Note",
        },
    )
    with st.expander(f"Order blotter: every fill sent to market ({len(sim['fills']):,})"):
        st.dataframe(
            sim["fills"], hide_index=True, width="stretch", height=300,
            column_config={
                "time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                "pair": "Pair", "side": "Side", "price": "Price", "reason": "Reason",
                "notional": st.column_config.NumberColumn("Notional ($)", format="%.0f"),
                "slippage": st.column_config.NumberColumn("Slippage ($)", format="%.2f"),
                "commission": st.column_config.NumberColumn("Transaction cost ($)", format="%.2f"),
            },
        )

    front = sim["frontier"]
    st.markdown("#### Efficient frontier at the latest rebalance that held a position")
    if front is None:
        st.info("No rebalance held a position, so there is no frontier to show.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=front["risks"] * 100, y=front["rewards"] * 100, mode="lines", name="Frontier",
                        line=dict(color=SLOTS[0], width=2),
                        hovertemplate="risk %{x:.3f}%<br>reward %{y:.4f}%<extra></extra>")
        fig.add_scatter(x=[front["risk"] * 100], y=[front["reward"] * 100], mode="markers", name="Chosen point",
                        marker=dict(color=SLOTS[1], size=12, line=dict(color=SURFACE, width=2)),
                        hovertemplate="chosen<br>risk %{x:.3f}%<br>reward %{y:.4f}%<extra></extra>")
        show(style(fig, height=320, ytitle="Expected return per trade", xtitle="Standard deviation per trade",
                   ysuffix="%", unified=False).update_xaxes(ticksuffix="%"))
        held_txt = ", ".join(f"{l} {w:.2f}×" for l, w in zip(front["labels"], front["weights"]) if w > 0)
        st.caption(f"{front['time']:%Y-%m-%d %H:%M} UTC, {len(front['labels'])} open signals. Held: {held_txt}.")

# ---- Methodology
def _dot(body: str, direction: str, ranksep: float) -> str:
    """Wrap diagram nodes and edges in shared Graphviz styling that follows the page theme."""
    return f"""
    digraph G {{
      rankdir={direction}; bgcolor="transparent"; nodesep=0.35; ranksep={ranksep};
      node [shape=box, style="rounded,filled", fillcolor="{SURFACE}", color="{AXIS}", penwidth=1.2,
            fontname="Arial", fontsize=12, fontcolor="{INK}", margin="0.18,0.10"];
      edge [color="{MUTED}", fontname="Arial", fontsize=11, fontcolor="{INK2}", arrowsize=0.7];
      {body}
    }}"""


with tab_method:
    every = f"every {fmt_hours(rebalance_hours)}"
    stop_on = trail_pct > 0
    decision = f'shape=diamond, style="filled", fillcolor="{SURFACE}", color="{SLOTS[0]}", penwidth=1.6, margin="0.05,0.05"'

    st.subheader("Methodology")
    st.write(
        "How the system works from start to finish. The numbers below come from the current sidebar "
        "settings, so this page always describes the simulation on the other tabs."
    )

    st.markdown("#### 1. Before trading: done once")
    show_dot = lambda src: st.graphviz_chart(src, width="content")  # noqa: E731
    show_dot(_dot(f"""
      data   [label="Hourly candles\\n{len(sel_pairs)} pairs,\\none time grid"];
      sig    [label="Strategies\\nlong / short / flat\\nsignals"];
      tune   [label="Tune\\nwalk-forward\\n+ K-fold"];
      freeze [label="Freeze\\nparameters\\n(per pair for\\nNelstein1)"];
      est    [label="Estimate\\nreturn and risk\\nper trade"];
      live   [label="Trade\\n(diagram 2)", color="{SLOTS[0]}", penwidth=1.6];
      data -> sig -> tune -> freeze -> est -> live;
    """, "LR", 0.3))

    st.markdown("#### 2. While trading: repeated on every hourly bar")
    stop_label = (f"trailing stop follows the\\nbest price, {trail_pct:g}% behind" if stop_on else "no stop-loss set")
    show_dot(_dot(f"""
      bar    [label="Hourly bar closes\\nstrategies update their signals"];
      exitq  [label="Open position:\\nexit signal, or\\nstop-loss hit?", {decision}];
      close  [label="Close it now\\nexit signal → next bar's open\\nstop-loss → inside the bar"];
      check  [label="Rebalance\\ncheck due?\\n({every})", {decision}];
      change [label="Signals opened\\nor closed, or leverage\\nabove the cap?", {decision}];
      opt    [label="Optimizer sets the weight of each open signal\\nleverage ≤ {lev_cap:g}× equity · each currency ≤ {expo_cap:g}×"];
      order  [label="One net order per pair, filled at this bar's open\\npays {commission_pct:g}% transaction cost + slippage"];
      hold   [label="Hold until the next bar\\ncarry accrues · {stop_label}\\nidle equity earns the cash rate"];
      rec    [label="Record equity, drawdown, trades and fills"];
      bar -> exitq;
      exitq -> close [label=" yes"];
      exitq -> check [label=" no"];
      close -> check;
      check -> change [label=" yes"];
      check -> hold [label=" no"];
      change -> opt [label=" yes"];
      change -> hold [label=" no"];
      opt -> order -> hold -> rec;
      rec -> bar [label=" next bar", style=dashed, constraint=false];
    """, "TB", 0.38))
    st.caption("Blue diamonds are decisions. A new signal waits for the next rebalance check before any "
               "money goes into it; an exit or a stop-loss never waits.")

    st.markdown("#### 3. The rules in detail")
    left, right = st.columns(2)
    with left:
        st.markdown("**Data**")
        st.markdown(
            f"- 1-hour open, high, low, close candles for {', '.join('/'.join(PAIRS[p]) for p in sel_pairs)}, "
            f"{first_day} to {last_day}.\n"
            "- Only hours present for every pair are kept, so all pairs share one clock.\n"
            "- Nothing is traded when the market has no bar (weekends, holidays)."
        )
        st.markdown("**Signals**")
        for key in sel_strategies:
            s = STRATEGIES[key]
            st.markdown(f"- *{s.name}* ({params_text(key)}): {s.description}")
        st.markdown("- A signal is computed only from completed bars, when a bar closes.")

        st.markdown("**Tuning and estimation**")
        st.markdown(
            f"- Tuning period {tune_range[0]} to {tune_range[1]}: each parameter set is scored by "
            f"{MEASURES[tune_measure][0]} using walk-forward ({int(train_months)}-month training window, "
            f"{int(test_months)}-month test block) and {int(n_folds)}-fold cross-validation.\n"
            "- One parameter set per strategy is frozen and never changed afterwards (Nelstein1: one per pair).\n"
            f"- Estimation period {est_range[0]} to {est_range[1]}: each strategy–pair's expected return is "
            "the mean of its per-trade returns and its risk is their standard deviation.\n"
            f"- The simulation then runs on later data, {sim_range[0]} to {sim_range[1]}."
        )

        st.markdown("**Investing: when money goes into a trade**")
        st.markdown(
            f"- A strategy opens a signal at a bar's close. The account acts on it at the next rebalance "
            f"check, at that bar's open.\n"
            "- If the signal has already closed by then, it is never traded.\n"
            "- Only signals whose strategy–pair has a positive expected return (after the cash interest "
            "given up on margin) can receive money.\n"
            f"- Position size is the optimizer's weight × equity. Objective: "
            f"{OBJECTIVES[objective][0].lower() + OBJECTIVES[objective][1:]}.\n"
            "- If two strategies point opposite ways on one pair, only the side with the higher expected "
            "return can be held."
        )
    with right:
        st.markdown("**Stop-losses**")
        if stop_on:
            st.markdown(
                f"- Every trade carries a trailing stop {trail_pct:g}% behind the best price since entry: the "
                "highest high for a long, the lowest low for a short.\n"
                "- The stop only moves in the trade's favour, so it locks in gains as price runs.\n"
                "- Inside a bar the stop level comes from earlier bars only; the bar's own high cannot "
                "rescue its low.\n"
                "- It fills at the stop price. If a bar opens beyond the stop (a weekend gap, for example), "
                "it fills at that open, which is worse.\n"
                "- After a stop the strategy stays flat on that pair until its signal changes."
            )
        else:
            st.markdown("- No stop-loss is set. Positions close only on the strategy's exit signal.")
        if target_pct > 0:
            st.markdown(f"- Take-profit {target_pct:g}% from entry, filled at its price. If one bar reaches "
                        "both stop and target, the stop is assumed to fill.")
        for k in sel_strategies:
            if STRATEGIES[k].own_stops:
                st.markdown(f"- *{STRATEGIES[k].name} sets its own stops* (the settings above apply to the "
                            f"other strategies): {STRATEGIES[k].stop_note}. They fill the same way.")

        st.markdown("**Rebalancing**")
        st.markdown(
            f"- **When it is checked:** {every} of clock time, counted from midnight UTC, at the open of the "
            "first bar at or after each scheduled time (the Rebalancing tab compares intervals).\n"
            "- **What triggers a rebalance at a check:** a signal opened or closed since the last check, "
            f"or total leverage drifted more than {min_trade_pct:g}% of equity above the {lev_cap:g}× cap. "
            "Otherwise nothing is traded.\n"
            "- **What it does:** re-optimizes the weights of all open signals together, then trades to them.\n"
            f"- **Limits:** total notional ≤ {lev_cap:g}× equity; net exposure to any one currency ≤ "
            f"{expo_cap:g}× equity; weights are never negative.\n"
            f"- **Small changes are skipped:** resizing a position by under {min_trade_pct:g}% of equity is "
            "not worth its cost.\n"
            "- **One net order per pair:** if two strategies hold the same pair, only the net change is traded.\n"
            "- **Exits are not rebalances:** an exit signal or stop-loss closes that position immediately; "
            "the rest of the book is resized at the next check."
        )

        st.markdown("**Costs on every fill**")
        st.markdown(
            f"- Transaction cost: {commission_pct:g}% of the traded notional.\n"
            f"- Slippage: {slip_mult:g} × the pair's standard deviation of hourly returns over the previous "
            f"{VOL_WINDOW} bars, so volatile hours cost more. Stop-loss fills pay it too.\n"
            "- A round trip (entry plus exit) therefore costs about "
            f"{2 * (np.mean([market.pairs[p].vol.mean() for p in sel_pairs]) * 1e4 * slip_mult + commission_pct * 100):.1f} "
            "basis points on average."
        )

        st.markdown("**The account**")
        st.markdown(
            f"- Margin account starting with {capital:,.0f} USD; margin requirement = 1 ÷ leverage cap.\n"
            f"- Equity not posted as margin earns {cash_pct:g}% a year.\n"
            "- Open positions earn or pay carry: the interest-rate gap between the pair's two currencies, "
            "for the time actually held, weekends included.\n"
            "- Profits earned in CAD or JPY are converted to USD at the current rate.\n"
            "- Equity = starting capital + price P&L + carry + interest − slippage − transaction costs, "
            "recorded every hour. Drawdown is the fall from the highest equity so far."
        )
    st.caption("Placeholders and parts not built yet are listed on the Assumptions tab.")


# ---- Strategy trades
with tab_bt:
    st.subheader("Strategy trades")
    c = st.columns(2)
    bt_key = c[0].selectbox("Strategy", sel_strategies, format_func=lambda k: STRATEGIES[k].name)
    bt_period = c[1].selectbox("Period", list(PERIODS), index=2)
    strat = STRATEGIES[bt_key]
    stops = strat.stop_note if strat.own_stops else f"a {trail_pct:g}% trailing stop"
    st.caption(
        f"{strat.description} Frozen parameters: {params_text(bt_key)}. Every trade is "
        f"{BACKTEST_NOTIONAL:,.0f} USD notional, with {stops}, "
        f"{commission_pct:g}% transaction cost per fill and volatility-scaled slippage."
    )
    runs = {p: c_backtest(bt_key, p, frozen[bt_key][p], rates, *PERIODS[bt_period], eng) for p in sel_pairs}

    fig = go.Figure()
    rows = []
    for p, (tr, hr) in runs.items():
        daily_eq = hr["equity"].resample("1D").last().dropna()
        fig.add_scatter(x=daily_eq.index, y=daily_eq, name=p, mode="lines",
                        line=dict(color=PAIR_COLOR[p], width=2),
                        hovertemplate="$%{y:,.0f}<extra>" + p + "</extra>")
        m = compute_measures(hr["ret"].to_numpy(), cash_rate=cash)
        s = trade_stats(tr["net_pnl"].to_numpy(), tr["hours_held"].to_numpy())
        rows.append({
            "Pair": p, "Trades": s["trades"], "Win rate (%)": s["win_rate"] * 100,
            "Profit factor": s["profit_factor"], "Net P&L ($)": tr["net_pnl"].sum(),
            "Return (%)": m["total_return"] * 100, "Max drawdown (%)": m["max_drawdown"] * 100,
            "Sharpe": m["sharpe"], "Costs ($)": tr["slippage"].sum() + tr["commission"].sum(),
            "Stopped out": int((tr["exit_reason"] == "trailing stop").sum()),
        })
    fig.add_hline(y=BACKTEST_NOTIONAL, line=dict(color=MUTED, width=1, dash="dot"))
    st.markdown(f"**Equity curve by pair, {strat.name}** (each pair starts at {BACKTEST_NOTIONAL:,.0f} USD)")
    show(style(fig, ytitle="USD", yprefix="$"))
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Win rate (%)": st.column_config.NumberColumn(format="%.0f"),
            "Profit factor": st.column_config.NumberColumn(format="%.2f"),
            "Net P&L ($)": st.column_config.NumberColumn(format="%.0f"),
            "Return (%)": st.column_config.NumberColumn(format="%.2f"),
            "Max drawdown (%)": st.column_config.NumberColumn(format="%.2f"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Costs ($)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    st.markdown("#### One pair, trade by trade")
    bt_pair = st.selectbox("Pair", sel_pairs)
    trades, bt_hourly = runs[bt_pair]
    if trades.empty:
        st.info("No trades in this period" + (f": {strat.no_trade_hint}." if strat.no_trade_hint else "."))
    else:
        _, _, _, mk = get_market()
        i0, i1 = mk.window(*PERIODS[bt_period])
        close = pd.Series(mk.pairs[bt_pair].close[i0:i1], index=mk.index[i0:i1])
        longs, shorts = trades[trades["direction"] > 0], trades[trades["direction"] < 0]
        stops = trades[trades["exit_reason"] == "trailing stop"]
        others = trades[trades["exit_reason"] != "trailing stop"]
        fig = go.Figure()
        fig.add_scatter(x=close.index, y=close, name="Close", mode="lines",
                        line=dict(color=MUTED, width=1.5), hovertemplate="%{y:.5f}<extra>Close</extra>")
        fig.add_scatter(x=bt_hourly.index, y=bt_hourly["stop"], name="Trailing stop", mode="lines",
                        line=dict(color=SLOTS[4], width=1.5, dash="dot"),
                        hovertemplate="%{y:.5f}<extra>Trailing stop</extra>")
        fig.add_scatter(x=longs["entry_time"], y=longs["entry_price"], name="Long entry", mode="markers",
                        marker=dict(symbol="triangle-up", size=11, color=SLOTS[2], line=dict(color=SURFACE, width=1.5)),
                        hovertemplate="%{y:.5f}<extra>Long entry</extra>")
        fig.add_scatter(x=shorts["entry_time"], y=shorts["entry_price"], name="Short entry", mode="markers",
                        marker=dict(symbol="triangle-down", size=11, color=SLOTS[1], line=dict(color=SURFACE, width=1.5)),
                        hovertemplate="%{y:.5f}<extra>Short entry</extra>")
        fig.add_scatter(x=others["exit_time"], y=others["exit_price"], name="Exit (signal)", mode="markers",
                        marker=dict(symbol="circle-open", size=9, color=INK, line=dict(width=2)),
                        hovertemplate="%{y:.5f}<extra>Exit (signal)</extra>")
        fig.add_scatter(x=stops["exit_time"], y=stops["exit_price"], name="Exit (trailing stop)", mode="markers",
                        marker=dict(symbol="x", size=9, color=INK),
                        hovertemplate="%{y:.5f}<extra>Exit (trailing stop)</extra>")
        st.markdown(f"**{bt_pair} price with entries and exits** (drag to zoom, double-click to reset)")
        show(style(fig, height=420, ytitle=bt_pair, unified=False))

        m = compute_measures(bt_hourly["ret"].to_numpy(), cash_rate=cash)
        measure_tiles(m, ["total_return", "annual_return", "std", "max_drawdown", "sharpe", "sortino", "calmar"])
        trade_tiles(trade_stats(trades["net_pnl"].to_numpy(), trades["hours_held"].to_numpy()))
        st.markdown(f"**Equity curve, {strat.name} on {bt_pair}**")
        show(equity_chart(bt_hourly.index, bt_hourly["equity"], PAIR_COLOR[bt_pair], BACKTEST_NOTIONAL))
        dd = drawdown_series(bt_hourly["equity"].to_numpy(), BACKTEST_NOTIONAL)
        st.markdown("**Drawdown**")
        show(drawdown_chart(bt_hourly.index, dd, PAIR_COLOR[bt_pair]))
        st.caption(drawdown_caption(bt_hourly.index, dd))

        c = st.columns(4)
        c[0].metric("Price P&L", money(trades["price_pnl"].sum()))
        c[1].metric("Carry", money(trades["carry"].sum()))
        c[2].metric("Slippage paid", money(trades["slippage"].sum()))
        c[3].metric("Transaction costs paid", money(trades["commission"].sum()))
        view = trades.drop(columns=["entry_i", "exit_i", "pair"]).copy()
        view["direction"] = view["direction"].map({1: "long", -1: "short"})
        view["ret_gross"] *= 100
        view["ret_net"] *= 100
        st.markdown("**Trade list**")
        st.dataframe(
            view, hide_index=True, width="stretch", height=320,
            column_config={
                "entry_time": st.column_config.DatetimeColumn("Entry", format="YYYY-MM-DD HH:mm"),
                "exit_time": st.column_config.DatetimeColumn("Exit", format="YYYY-MM-DD HH:mm"),
                "direction": "Side", "entry_price": "Entry price", "exit_price": "Exit price",
                "exit_reason": "Exit reason",
                "hours_held": st.column_config.NumberColumn("Hours held", format="%.0f"),
                "price_pnl": st.column_config.NumberColumn("Price P&L ($)", format="%.0f"),
                "carry": st.column_config.NumberColumn("Carry ($)", format="%.1f"),
                "slippage": st.column_config.NumberColumn("Slippage ($)", format="%.1f"),
                "commission": st.column_config.NumberColumn("Transaction cost ($)", format="%.1f"),
                "net_pnl": st.column_config.NumberColumn("Net P&L ($)", format="%.0f"),
                "ret_gross": st.column_config.NumberColumn("Return before costs (%)", format="%.3f"),
                "ret_net": st.column_config.NumberColumn("Net return (%)", format="%.3f"),
            },
        )

# ---- Nelstein1 study
def _fmt(v, kind="sharpe"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if kind == "sharpe":
        return f"{v:+.2f}"
    if kind == "p":
        return f"{v:.2f}"
    if kind == "pct":
        return f"{v:.0f}%"
    if kind == "money":
        return money(v)
    if kind == "int":
        return f"{int(v)}"
    return str(v)


with tab_nel:
    st.subheader("Nelstein1 study")
    st.write(
        "Is Nelstein1 worth trading? Each pair is tuned on the tuning period "
        f"({tune_range[0]} to {tune_range[1]}) and then judged on the simulation period "
        f"({sim_range[0]} to {sim_range[1]}), which tuning never saw: the *holdout*. Sharpe ratios here are on "
        "the trade notional with no cash rate subtracted, so zero means no edge. All runs pay the sidebar costs."
    )
    S = c_study(tuple(sel_pairs), rates, tune_range, sim_range, eng, tune_measure,
                int(train_months), int(test_months), int(n_folds), int(embargo), cash)
    T = S["table"].set_index("Pair")
    comb = S["combined"]
    prem = S["premises"]
    n = len(T)
    n_net = int((T["Holdout Sharpe"] > 0).sum())
    n_gross = int((T["Gross Sharpe"] > 0).sum())
    n_sig = int((T["Random-walk p-value"] <= 0.10).sum())
    right = prem["Right direction (%)"]
    no_edge = comb["net"] <= 0 and n_sig == 0

    st.markdown("#### Verdict")
    verdict = [
        f"**After costs:** {n_net} of {n} pairs made money on the holdout. All {n} together at equal size: "
        f"Sharpe {comb['net']:+.2f}.",
        f"**Before costs:** Sharpe {comb['gross']:+.2f}, with {n_gross} of {n} pairs positive"
        + (", so the losses are not just trading costs: the entries themselves have no edge."
           if comb["gross"] <= 0 else ", so there is some edge before costs, and costs eat it."),
        f"**Against chance:** {n_sig} of {n} pairs did better than the same rules run on {N_SYNTHETIC} "
        "shuffled, trendless versions of their own data (10% level).",
        f"**The signals themselves:** after a trend or breakout signal, the next day's move went the signal's way "
        f"{right.min():.0f}–{right.max():.0f}% of the time, where a coin flip gives 50%.",
        f"**Look-ahead:** acting one bar later gives Sharpe {comb['delay']:+.2f}, close to {comb['net']:+.2f}, "
        "so the results do not lean on information the strategy could not have had.",
    ]
    box = st.warning if no_edge else st.info
    box(("As specified and tuned, Nelstein1 shows no tradable edge on this data.\n\n" if no_edge else "")
        + "\n".join(f"- {line}" for line in verdict), icon="⚠️" if no_edge else "ℹ️")

    # 1. Performance
    st.markdown("#### 1. How it performs")
    c = st.columns(4)
    c[0].metric("Holdout Sharpe, all pairs", f"{comb['net']:+.2f}")
    c[1].metric("Before costs", f"{comb['gross']:+.2f}")
    c[2].metric("Holdout trades", f"{int(T['Trades'].sum()):,}")
    c[3].metric("Pairs profitable", f"{n_net} of {n}")
    perf = T[["Parameters", "In-sample Sharpe", "Walk-forward OOS Sharpe", "Holdout Sharpe", "Holdout return (%)",
              "Holdout max drawdown (%)", "Trades", "Win rate (%)", "Profit factor"]].reset_index()
    st.dataframe(perf, hide_index=True, width="stretch", column_config={
        "Parameters": "Frozen parameters",
        "In-sample Sharpe": st.column_config.NumberColumn("Sharpe, tuning period", format="%.2f"),
        "Walk-forward OOS Sharpe": st.column_config.NumberColumn("Sharpe, walk-forward OOS", format="%.2f"),
        "Holdout Sharpe": st.column_config.NumberColumn("Sharpe, holdout", format="%.2f"),
        "Holdout return (%)": st.column_config.NumberColumn("Holdout return (%)", format="%.2f"),
        "Holdout max drawdown (%)": st.column_config.NumberColumn("Holdout max drawdown (%)", format="%.2f"),
        "Win rate (%)": st.column_config.NumberColumn(format="%.0f"),
        "Profit factor": st.column_config.NumberColumn(format="%.2f"),
    })
    st.caption("Reading across: the tuning-period Sharpe is what the tuner saw, walk-forward re-tunes on a rolling "
               "window inside the tuning period, and the holdout is the real test.")
    fig = go.Figure()
    for p in sel_pairs:
        eq = S["equity"][p].resample("1D").last().dropna()
        fig.add_scatter(x=eq.index, y=eq, name=p, mode="lines", line=dict(color=PAIR_COLOR[p], width=2),
                        hovertemplate="$%{y:,.0f}<extra>" + p + "</extra>")
    fig.add_hline(y=BACKTEST_NOTIONAL, line=dict(color=MUTED, width=1, dash="dot"))
    st.markdown(f"**Holdout equity curve by pair** (each starts at {BACKTEST_NOTIONAL:,.0f} USD, after costs)")
    show(style(fig, ytitle="USD", yprefix="$"))

    # 2. Your own settings
    st.markdown("#### 2. Try your own settings")
    st.write(
        "Set x, y, w, z and the direction, then click a **Run** button. Nothing runs while you change the "
        "values. Use the top row to give every pair the same settings, or edit pairs one by one in the table."
    )
    st.caption(
        "Direction: *As specified* trades breakouts and trends as described. *Reversed* flips every direction. "
        "*S1 faded* bets against range breakouts but still follows trends. Settings chosen by hand after looking "
        "at the holdout are fitted to it, so check the tuning-period column as well."
    )
    ss = st.session_state
    ss.setdefault("nel_ver", 0)
    tuned_rows = pd.DataFrame({
        "Pair": list(T.index), "x": T["x"].astype(int).to_numpy(), "y": T["y"].astype(float).to_numpy(),
        "w": T["w"].astype(float).to_numpy(), "z": T["z"].astype(float).to_numpy(), "Direction": "As specified",
    })
    base_rows = ss.get("nel_base")
    if base_rows is None or list(base_rows["Pair"]) != list(tuned_rows["Pair"]):
        base_rows = tuned_rows
    ss.nel_base_current = base_rows.reset_index(drop=True)

    def _nel_commit(df: pd.DataFrame) -> None:
        df = df.reset_index(drop=True)
        ss.nel_base = df
        ss.nel_applied = tuple((r.Pair, int(r.x), float(r.y), float(r.w), float(r.z), str(r.Direction))
                               for r in df.itertuples(index=False))
        ss.nel_apply_portfolio = bool(ss.get("nel_use_portfolio", True))
        ss.nel_ver += 1   # a fresh table, showing exactly what ran

    def _nel_run_all() -> None:
        df = ss.nel_base_current.copy()
        df["x"], df["y"], df["w"], df["z"] = int(ss.nel_qx), float(ss.nel_qy), float(ss.nel_qw), float(ss.nel_qz)
        df["Direction"] = ss.nel_qd
        _nel_commit(df)

    def _nel_run_table() -> None:
        df = ss.nel_base_current.copy()
        edits = ss.get(f"nel_editor_{ss.nel_ver}") or {}
        for i, changes in (edits.get("edited_rows") or {}).items():
            for col, val in changes.items():
                if col in df.columns and val is not None:
                    df.at[int(i), col] = val
        _nel_commit(df)

    def _nel_reset() -> None:
        for k in ("nel_base", "nel_applied", "nel_apply_portfolio"):
            ss.pop(k, None)
        ss.nel_ver += 1

    with st.form("nel_custom_form"):
        st.markdown("**Same settings for every pair**")
        q = st.columns([1, 1, 1, 1, 1.3, 1.4], vertical_alignment="bottom")
        q[0].number_input("x: lookback (bars)", 6, 1000, 72, 6, key="nel_qx")
        q[1].number_input("y: band", 0.1, 10.0, 2.0, 0.25, key="nel_qy")
        q[2].number_input("w: trend stop", 0.1, 50.0, 4.0, 0.5, key="nel_qw")
        q[3].number_input("z: breakout stop (%)", 0.05, 10.0, 1.0, 0.25, key="nel_qz")
        q[4].selectbox("Direction", list(DIRECTIONS), key="nel_qd")
        q[5].form_submit_button("Run for every pair", on_click=_nel_run_all, width="stretch")

        st.markdown("**Or set each pair separately**")
        st.data_editor(
            ss.nel_base_current, key=f"nel_editor_{ss.nel_ver}", hide_index=True, width="stretch",
            disabled=["Pair"],
            column_config={
                "x": st.column_config.NumberColumn("x: lookback (bars)", min_value=6, max_value=1000, step=1,
                                                   format="%d", required=True),
                "y": st.column_config.NumberColumn("y: band (× average reversion)", min_value=0.1, max_value=10.0,
                                                   step=0.05, format="%.2f", required=True),
                "w": st.column_config.NumberColumn("w: trend stop (× average pullback)", min_value=0.1,
                                                   max_value=50.0, step=0.1, format="%.2f", required=True),
                "z": st.column_config.NumberColumn("z: breakout stop (%)", min_value=0.05, max_value=10.0,
                                                   step=0.05, format="%.2f", required=True),
                "Direction": st.column_config.SelectboxColumn("Direction", options=list(DIRECTIONS), required=True),
            },
        )
        st.checkbox("Also use these settings for Nelstein1 in the portfolio and the other tabs", value=True,
                    key="nel_use_portfolio")
        b = st.columns([1.3, 1.3, 3.4])
        b[0].form_submit_button("Run the table", type="primary", on_click=_nel_run_table, width="stretch")
        b[1].form_submit_button("Reset to tuned", on_click=_nel_reset, width="stretch")

    applied = ss.get("nel_applied")
    if applied and [r[0] for r in applied] == list(T.index):
        C = c_custom(applied, rates, tune_range, sim_range, eng)
        ct = C["table"].copy()
        ct.insert(5, "Tuned holdout Sharpe", T.loc[ct["Pair"], "Holdout Sharpe"].to_numpy())
        if ss.get("nel_apply_portfolio"):
            st.success("These settings are now driving Nelstein1 in the portfolio, estimation and strategy-trades "
                       "tabs. **Reset to tuned** puts the tuned settings back.", icon="✅")
        c = st.columns(3)
        c[0].metric("Holdout Sharpe, all pairs", f"{C['combined']:+.2f}",
                    delta=f"{C['combined'] - comb['net']:+.2f} vs tuned", delta_color="normal")
        c[1].metric("Pairs profitable", f"{int((ct['Holdout Sharpe'] > 0).sum())} of {len(ct)}")
        c[2].metric("Holdout trades", f"{int(ct['Trades'].sum()):,}")
        st.dataframe(ct, hide_index=True, width="stretch", column_config={
            "Sharpe, tuning period": st.column_config.NumberColumn(format="%.2f"),
            "Holdout Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Tuned holdout Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Holdout return (%)": st.column_config.NumberColumn(format="%.2f"),
            "Holdout max drawdown (%)": st.column_config.NumberColumn(format="%.2f"),
            "Win rate (%)": st.column_config.NumberColumn(format="%.0f"),
            "Profit factor": st.column_config.NumberColumn(format="%.2f"),
            "Gross Sharpe": st.column_config.NumberColumn("Sharpe before costs", format="%.2f"),
        })
        fig = go.Figure()
        for pp, eq in C["equity"].items():
            eqd = eq.resample("1D").last().dropna()
            fig.add_scatter(x=eqd.index, y=eqd, name=pp, mode="lines", line=dict(color=PAIR_COLOR[pp], width=2),
                            hovertemplate="$%{y:,.0f}<extra>" + pp + "</extra>")
        fig.add_hline(y=BACKTEST_NOTIONAL, line=dict(color=MUTED, width=1, dash="dot"))
        st.markdown("**Holdout equity curve with these settings**")
        st.plotly_chart(style(fig, height=300, ytitle="USD", yprefix="$"), width="stretch",
                        config={"displayModeBar": False}, key=f"nel_custom_equity_{hash(applied)}")
    else:
        st.info("Set the values, then click **Run for every pair** or **Run the table**.")

    # 3. Best y and w with x and z fixed
    st.markdown("#### 3. Best y and w with x and z held fixed")
    st.write(
        f"For each pair, x and z stay fixed while every combination of y ({len(Y_GRID)} values, "
        f"{Y_GRID[0]:g}–{Y_GRID[-1]:g}) and w ({len(W_GRID)} values, {W_GRID[0]:g}–{W_GRID[-1]:g}) is backtested. "
        "The best pair of values is picked on the tuning period, then checked on the holdout. The holdout's own "
        "best is shown too, but it could only be found by looking at the holdout, so it is hindsight."
    )
    with st.form("nel_yw_form"):
        f = st.columns([1.6, 0.8, 0.8, 1.5, 1.2, 1.2], vertical_alignment="bottom")
        yw_src = f[0].radio("x and z", ["Each pair's tuned values", "Same for every pair"], key="nel_yw_src")
        yw_x = f[1].number_input("x (bars)", 6, 1000, 72, 6, key="nel_yw_x")
        yw_z = f[2].number_input("z (%)", 0.05, 10.0, 1.0, 0.25, key="nel_yw_z")
        yw_obj = f[3].selectbox("Best means highest", list(SEARCH_OBJECTIVES),
                                format_func=lambda k: SEARCH_OBJECTIVES[k], key="nel_yw_obj")
        yw_dir = f[4].selectbox("Direction", list(DIRECTIONS), key="nel_yw_dir")
        yw_go = f[5].form_submit_button("Search y and w", type="primary", width="stretch")
    if yw_go:
        ss.nel_yw_request = (yw_src, int(yw_x), float(yw_z), yw_obj, yw_dir)
    req = ss.get("nel_yw_request")
    if req is None:
        st.info("Choose where x and z come from and what *best* means, then click **Search y and w**.")
    else:
        src, fx, fz, obj, fdir = req
        setups = tuple((pp, int(T.loc[pp, "x"]), float(T.loc[pp, "z"]), fdir) if src.startswith("Each")
                       else (pp, fx, fz, fdir) for pp in T.index)
        Y = c_yw(setups, rates, tune_range, sim_range, eng, obj)
        sm = Y["summary"]
        tuned_ret = T["Holdout return (%)"]
        c = st.columns(4)
        c[0].metric("Average holdout return, picked y and w", f"{sm['Holdout return (%)'].mean():+.2f}%",
                    delta=f"{sm['Holdout return (%)'].mean() - tuned_ret.mean():+.2f} pts vs tuned", delta_color="normal")
        c[1].metric("Pairs positive on the holdout", f"{int((sm['Holdout return (%)'] > 0).sum())} of {len(sm)}")
        c[2].metric("Average holdout return, hindsight best", f"{sm['Holdout-best return (%)'].mean():+.2f}%")
        c[3].metric("Settings positive on the holdout", f"{sm['Settings positive on holdout (%)'].mean():.0f}%")
        st.dataframe(sm, hide_index=True, width="stretch", column_config={
            "z (fixed)": st.column_config.NumberColumn(format="%.2f"),
            "Best y": st.column_config.NumberColumn("Best y (tuning)", format="%.2f"),
            "Best w": st.column_config.NumberColumn("Best w (tuning)", format="%.1f"),
            "Tuning-period return (%)": st.column_config.NumberColumn(format="%.2f"),
            "Tuning-period Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Holdout return (%)": st.column_config.NumberColumn(format="%.2f"),
            "Holdout Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Holdout max drawdown (%)": st.column_config.NumberColumn(format="%.2f"),
            "Holdout-best y": st.column_config.NumberColumn("Hindsight best y", format="%.2f"),
            "Holdout-best w": st.column_config.NumberColumn("Hindsight best w", format="%.1f"),
            "Holdout-best return (%)": st.column_config.NumberColumn("Hindsight best return (%)", format="%.2f"),
            "Settings positive on holdout (%)": st.column_config.NumberColumn(format="%.0f"),
            "Rank correlation tuning vs holdout": st.column_config.NumberColumn(format="%.2f"),
        })

        def _load_yw() -> None:
            df = pd.DataFrame({"Pair": sm["Pair"], "x": sm["x (fixed)"].astype(int), "y": sm["Best y"],
                               "w": sm["Best w"], "z": sm["z (fixed)"], "Direction": sm["Direction"]})
            _nel_commit(df)

        st.button("Load these y and w into section 2 and run them", on_click=_load_yw, key="nel_yw_load",
                  help="Fills the settings table above with the picked values and runs them; with the box there "
                       "ticked, they also drive the portfolio.")

        fig = go.Figure()
        for pp, eq in Y["equity"].items():
            eqd = eq.resample("1D").last().dropna()
            fig.add_scatter(x=eqd.index, y=eqd, name=pp, mode="lines", line=dict(color=PAIR_COLOR[pp], width=2),
                            hovertemplate="$%{y:,.0f}<extra>" + pp + "</extra>")
        fig.add_hline(y=BACKTEST_NOTIONAL, line=dict(color=MUTED, width=1, dash="dot"))
        st.markdown("**Holdout equity curve with the picked y and w**")
        st.plotly_chart(style(fig, height=300, ytitle="USD", yprefix="$"), width="stretch",
                        config={"displayModeBar": False}, key=f"nel_yw_equity_{hash(req)}")

        hp_ = st.selectbox("Map for pair", list(T.index), key="nel_yw_pair")
        g = Y["grids"][hp_]
        row = sm.set_index("Pair").loc[hp_]
        scale = 100.0 if obj == "total_return" else 1.0
        unit = "%" if obj == "total_return" else ""
        left, right = st.columns(2)
        for col, prefix, title, best_y, best_w in (
            (left, "tune", "Tuning period", row["Best y"], row["Best w"]),
            (right, "hold", "Holdout", row["Holdout-best y"], row["Holdout-best w"]),
        ):
            zz = g.pivot_table(index="y", columns="w", values=f"{prefix}_{obj}").reindex(index=list(Y_GRID),
                                                                                       columns=list(W_GRID)) * scale
            lim = float(np.nanmax(np.abs(zz.to_numpy()))) or 1.0
            heat = go.Figure(go.Heatmap(
                z=zz.to_numpy(), x=[f"{v:g}" for v in W_GRID], y=[f"{v:g}" for v in Y_GRID], zmid=0,
                zmin=-lim, zmax=lim, xgap=1, ygap=1,
                colorscale=[[0, "#e34948"], [0.5, "#383835" if DARK else "#f0efec"], [1, SLOTS[0]]],
                colorbar=dict(title=SEARCH_OBJECTIVES[obj].split(" (")[0], thickness=10, ticksuffix=unit),
                hovertemplate="y=%{y}, w=%{x}<br>%{z:.2f}" + unit + "<extra></extra>",
            ))
            for by, bw, dash in ((row["Best y"], row["Best w"], "solid"), (best_y, best_w, "dot")):
                xi, yi = list(W_GRID).index(bw), list(Y_GRID).index(by)
                heat.add_shape(type="rect", x0=xi - 0.5, x1=xi + 0.5, y0=yi - 0.5, y1=yi + 0.5,
                               line=dict(color=INK, width=2, dash=dash))
            heat = style(heat, height=420, xtitle="w: trend stop (× average pullback)",
                         ytitle="y: band (× average reversion)", unified=False)
            heat.update_xaxes(type="category")   # equal-width cells; outlines use cell positions
            heat.update_yaxes(type="category")
            with col:
                st.markdown(f"**{title}**")
                show(heat)
        st.caption(
            f"{hp_}, x={int(row['x (fixed)'])}, z={row['z (fixed)']:g}%. Solid outline: the pick from the tuning "
            "period. Dotted outline: the holdout's own best (hindsight). Blue is positive, red negative."
        )
        edge = int(((sm["Best y"] == Y_GRID[0]) | (sm["Best w"] == W_GRID[-1]) | (sm["Best w"] >= 15)).sum())
        st.caption(
            f"Reading it: in {edge} of {len(sm)} pairs the tuning pick sits at a tiny band or a very wide trend stop, "
            "i.e. the search pushes Nelstein1 toward entering on almost any move and then holding the trend for "
            "a long time. The hindsight bests land in different places from the tuning picks, which is what "
            "noise looks like."
        )

    # 4. State machine
    st.markdown("#### 4. Does the state machine do what it says?")
    states = S["states"]
    if not states.empty:
        st.dataframe(states, hide_index=True, width="stretch", column_config={
            "Win rate (%)": st.column_config.NumberColumn(format="%.0f"),
            "Net P&L ($)": st.column_config.NumberColumn(format="%.0f"),
            "Gross P&L ($)": st.column_config.NumberColumn(format="%.0f"),
            "Costs ($)": st.column_config.NumberColumn(format="%.0f"),
            "Average hours held": st.column_config.NumberColumn(format="%.0f"),
        })
        by = states.set_index("Path")
        s1_rows = [r for r in by.index if r.startswith("S1")]
        s1_net = by.loc[s1_rows, "Net P&L ($)"].sum()
        s1_n = int(by.loc[s1_rows, "Trades"].sum())
        s1_to_s2 = int(by.loc["S1 breakout that turned into S2", "Trades"]) if "S1 breakout that turned into S2" in by.index else 0
        stayed = "S1 breakout that stayed S1"
        stayed_win = by.loc[stayed, "Win rate (%)"] if stayed in by.index else float("nan")
        st.caption(
            f"All pairs, holdout. Careful reading the S1 rows: a breakout only *turns into* S2 if price kept going "
            f"its way, so that split is decided by the outcome. It shows where the money went, not which entries to "
            f"keep. Judged as entries, the {s1_n:,} S1 breakouts {'lost' if s1_net < 0 else 'made'} "
            f"{money(abs(s1_net))} in total. {s1_to_s2 / max(s1_n, 1) * 100:.0f}% of them handed over to the S2 stop; "
            f"the z% stop decided the rest, and those won {stayed_win:.0f}% of the time."
        )

    st.markdown("**Price, band and entries on one pair (holdout)**")
    sp = st.selectbox("Pair", sel_pairs, key="nel_state_pair")
    row = T.loc[sp]
    _, _, _, mk_ = get_market()
    i0_, i1_ = mk_.window(*sim_range)
    trend_, bmean_, bmad_, _, _ = features_cached(mk_.pairs[sp].close, int(row["x"]))
    idx_ = mk_.index[i0_:i1_]
    close_ = mk_.pairs[sp].close[i0_:i1_]
    ranging = trend_[i0_:i1_] == 0
    upper_ = np.where(ranging, (bmean_ + row["y"] * bmad_)[i0_:i1_], np.nan)
    lower_ = np.where(ranging, (bmean_ - row["y"] * bmad_)[i0_:i1_], np.nan)
    tr_ = S["trades"][sp]
    fig = go.Figure()
    fig.add_scatter(x=idx_, y=close_, name="Close", mode="lines", line=dict(color=MUTED, width=1.5),
                    hovertemplate="%{y:.5f}<extra>Close</extra>")
    fig.add_scatter(x=idx_, y=upper_, name="S1 band", mode="lines", line=dict(color=SLOTS[0], width=1, dash="dot"),
                    hovertemplate="%{y:.5f}<extra>Band top</extra>")
    fig.add_scatter(x=idx_, y=lower_, name="S1 band", mode="lines", showlegend=False,
                    line=dict(color=SLOTS[0], width=1, dash="dot"), hovertemplate="%{y:.5f}<extra>Band bottom</extra>")
    for label, mask, symbol, color in (
        ("S1 long", (tr_["entry_state"] == "S1") & (tr_["direction"] > 0), "triangle-up", SLOTS[2]),
        ("S1 short", (tr_["entry_state"] == "S1") & (tr_["direction"] < 0), "triangle-down", SLOTS[1]),
        ("S2 entry", tr_["entry_state"] == "S2", "diamond", SLOTS[3]),
    ):
        g = tr_[mask]
        fig.add_scatter(x=g["entry_time"], y=g["entry_price"], name=label, mode="markers",
                        marker=dict(symbol=symbol, size=10, color=color, line=dict(color=SURFACE, width=1.5)),
                        hovertemplate="%{y:.5f}<extra>" + label + "</extra>")
    fig.add_scatter(x=tr_["exit_time"], y=tr_["exit_price"], name="Exit", mode="markers",
                    marker=dict(symbol="x", size=8, color=INK), hovertemplate="%{y:.5f}<extra>Exit</extra>")
    show(style(fig, height=420, ytitle=sp, unified=False))
    st.caption(f"{sp}: {row['Parameters']}. The band is drawn only while the market counts as ranging. "
               "Drag to zoom, double-click to reset.")

    # 3. Bias and error checks
    st.markdown("#### 5. Bias and error checks")
    pair_cols = list(T.index)

    def check(name, what, col, kind="sharpe", all_val=None, reading=""):
        r = {"Check": name, "What it tests": what}
        for pp in pair_cols:
            r[pp] = _fmt(T.loc[pp, col], kind)
        r["All pairs"] = _fmt(all_val, kind) if all_val is not None else ""
        r["Reading"] = reading
        return r

    rho = T["Rank correlation tuning vs holdout"]
    checks = [
        check("As specified", "Holdout Sharpe after costs", "Holdout Sharpe", all_val=comb["net"],
              reading="The baseline every other row is compared with."),
        check("Before costs", "Same trades, no slippage or transaction cost", "Gross Sharpe", all_val=comb["gross"],
              reading="Negative even before costs means no edge to protect." if comb["gross"] <= 0
              else "Positive before costs: the edge is smaller than the costs."),
        check("Costs doubled", "Slippage and transaction cost × 2", "Sharpe, costs doubled", all_val=comb["double"],
              reading="How fragile the result is to worse fills."),
        check("One bar late", "Every decision filled one bar later (look-ahead check)", "Sharpe, one bar late",
              all_val=comb["delay"],
              reading=("Close to the baseline: no sign of look-ahead." if abs(comb["delay"] - comb["net"]) < 0.5
                       else "Large change: check for information leaking from the future.")),
        check("Random-walk benchmark", f"Share of {N_SYNTHETIC} shuffled, trendless price paths that did as well "
              "(p-value)", "Random-walk p-value", kind="p",
              reading="Above 0.10: the real result is no better than chance."),
        check("Random-walk median", "Median Sharpe of the same rules on those paths", "Random-walk median Sharpe",
              reading="What luck alone produces with these rules and costs."),
        check("Reversed", "Every direction flipped: fade breakouts and trends", "Sharpe, reversed",
              all_val=comb["reverse"], reading="If this were strongly positive the signals would be backwards."),
        check("S1 faded", "Range breakouts faded, trends still followed", "Sharpe, S1 breakouts faded",
              all_val=comb["fade_s1"],
              reading=("Better than as specified: in a ranging market, betting on reversion beats betting on breakouts."
                       if comb["fade_s1"] > comb["net"] else "Not better: fading breakouts does not help either.")),
        check(f"Trend cut-off {TREND_K - 0.5:g}", "Hidden fixed setting loosened", "Sharpe, trend cut-off low",
              all_val=comb["k_low"], reading="A fixed setting that was never tuned."),
        check(f"Trend cut-off {TREND_K + 0.5:g}", "Hidden fixed setting tightened", "Sharpe, trend cut-off high",
              all_val=comb["k_high"], reading="Large swings either way mean the result rests on an arbitrary choice."),
        check("Tuning vs holdout", f"Rank correlation across all {len(STRATEGIES[STUDY_KEY].grid)} settings",
              "Rank correlation tuning vs holdout",
              reading=("Settings that ranked well in tuning also ranked well later, but compare the grid median below."
                       if rho.median() > 0.3 else "Tuning ranks barely predict holdout ranks.")),
        check("Grid median", "Median holdout Sharpe of all settings", "Grid median holdout Sharpe",
              reading="Negative: most settings lose, so ranking them only finds the least bad."),
        check("Chosen setting's rank", "Share of settings the frozen one beat on the holdout",
              "Chosen setting's holdout rank (%)", kind="pct", reading="50% is what picking at random gives."),
        check("Walk-forward OOS", "Sharpe of re-tuning on a rolling window", "Walk-forward OOS Sharpe",
              reading="Out-of-sample inside the tuning period."),
        check("Walk-forward choices", "Different settings chosen across the rolling windows",
              "WF parameter sets used", kind="int", reading="Many different choices means the best setting is unstable."),
        check("Profit concentration", "Top 5 trades' share of all winning P&L",
              "Top 5 trades' share of profits (%)", kind="pct", reading="High means a few lucky trades carry the result."),
        check("Long side", "Net P&L of long trades", "Long P&L ($)", kind="money", reading="Directional bias check."),
        check("Short side", "Net P&L of short trades", "Short P&L ($)", kind="money",
              reading="Losses on both sides: not a USD-trend artefact."),
    ]
    st.table(pd.DataFrame(checks).set_index("Check"))

    st.markdown("**Do the signals predict the next day at all?**")
    st.dataframe(prem, hide_index=True, width="stretch", column_config={
        "Average next-day move in signal direction (bp)": st.column_config.NumberColumn(format="%.2f"),
        "Right direction (%)": st.column_config.NumberColumn(format="%.1f"),
        "t-statistic (approx.)": st.column_config.NumberColumn(format="%.2f"),
    })
    st.caption(
        "For every bar each signal fires, the move from the next open to 24 bars later, signed by the signal's "
        "direction, over all five years and before stops or costs. A round trip costs about 3 bp. A |t| under 2 "
        "means no reliable direction."
    )

    # 5. Reversal on one pair
    st.markdown("#### 6. Reversing the strategy on one pair")
    st.write(
        "If a pair keeps losing, the signals may point the wrong way there. The *re-tuned* versions are tuned on "
        "the tuning period with the reversed rules before they see the holdout, so they are a fair test. "
        "*Reversed, same settings* flips the tuned settings because the holdout lost. That choice used the "
        "holdout, so its result is hindsight, not evidence."
    )
    rp = st.selectbox("Pair", sel_pairs, index=sel_pairs.index("USDJPY") if "USDJPY" in sel_pairs else 0,
                      key="nel_rev_pair")
    Rv = c_reversal(rp, rates, tune_range, sim_range, eng, tune_measure, int(train_months), int(test_months),
                    int(n_folds), int(embargo), cash)
    rt = Rv["table"]
    st.dataframe(rt, hide_index=True, width="stretch", column_config={
        "Sharpe, tuning period": st.column_config.NumberColumn(format="%.2f"),
        "Walk-forward OOS Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Holdout Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Holdout return (%)": st.column_config.NumberColumn(format="%.2f"),
        "Holdout max drawdown (%)": st.column_config.NumberColumn(format="%.2f"),
        "Win rate (%)": st.column_config.NumberColumn(format="%.0f"),
        "Profit factor": st.column_config.NumberColumn(format="%.2f"),
        "Gross Sharpe": st.column_config.NumberColumn("Sharpe before costs", format="%.2f"),
        "Random-walk p-value": st.column_config.NumberColumn(format="%.2f"),
    })
    fig = go.Figure()
    for k_, (name, eq) in enumerate(Rv["equity"].items()):
        eqd = eq.resample("1D").last().dropna()
        fig.add_scatter(x=eqd.index, y=eqd, name=name, mode="lines",
                        line=dict(color=SLOTS[k_], width=2, dash="dash" if "same settings" in name else "solid"),
                        hovertemplate="$%{y:,.0f}<extra>" + name + "</extra>")
    fig.add_hline(y=BACKTEST_NOTIONAL, line=dict(color=MUTED, width=1, dash="dot"))
    st.markdown(f"**{rp} holdout equity, as specified against reversed** (dashed = hindsight)")
    show(style(fig, height=320, ytitle="USD", yprefix="$"))
    fair = rt[rt["Holdout is out-of-sample"] == "yes"].set_index("Version")
    base_s = fair.loc["As specified, tuned", "Holdout Sharpe"]
    alt = fair.drop(index="As specified, tuned")
    best_name = alt["Holdout Sharpe"].idxmax()
    best_s = alt.loc[best_name, "Holdout Sharpe"]
    rev_gross = rt.set_index("Version")["Gross Sharpe"]
    if (rt["Holdout Sharpe"] <= 0).all():
        msg = (f"No version made money on {rp}. The best fair alternative, *{best_name}*, scores {best_s:+.2f} "
               f"against {base_s:+.2f} as specified, so reversing loses less but does not turn a profit. "
               + ("Both directions lose even before costs, so the problem is not which way the signals point: the "
                  "entries carry no information and the stops and costs do the damage."
                  if (rev_gross <= 0.1).all() else "Before costs the reversed rules come closer to break-even."))
    elif best_s > 0 >= base_s:
        msg = (f"*{best_name}* made money on {rp} ({best_s:+.2f}) where the specified rules lost ({base_s:+.2f}). "
               "Check its random-walk p-value and walk-forward result before trusting it.")
    else:
        msg = f"Best fair version: *{best_name}* at {best_s:+.2f}, against {base_s:+.2f} as specified."
    st.caption(msg)

    # 6. Parameter sensitivity
    st.markdown("#### 7. Parameter sensitivity")
    hp = st.selectbox("Pair", sel_pairs, key="nel_heat_pair")
    g = S["grids"][hp]
    chosen = T.loc[hp]
    xs, ws = sorted(g["x"].unique()), sorted(g["w"].unique())
    left, right = st.columns(2)
    for col, value, title in ((left, "Tuning score", "Tuning period (what the tuner saw)"),
                              (right, "Holdout Sharpe", "Holdout")):
        z = g.pivot_table(index="x", columns="w", values=value, aggfunc="median").reindex(index=xs, columns=ws)
        lim = float(np.nanmax(np.abs(z.to_numpy()))) or 1.0
        fig = go.Figure(go.Heatmap(
            z=z.to_numpy(), x=[f"w={v:g}" for v in ws], y=[f"x={v:g}" for v in xs], zmid=0, zmin=-lim, zmax=lim,
            colorscale=[[0, "#e34948"], [0.5, "#383835" if DARK else "#f0efec"], [1, SLOTS[0]]],
            text=[[f"{v:+.2f}" for v in r] for r in z.to_numpy()], texttemplate="%{text}",
            textfont=dict(color=INK), xgap=2, ygap=2, colorbar=dict(title="Sharpe", thickness=10),
            hovertemplate="%{y}, %{x}<br>median Sharpe %{z:.2f}<extra></extra>",
        ))
        fig.add_shape(type="rect", xref="x", yref="y",
                      x0=ws.index(chosen["w"]) - 0.5, x1=ws.index(chosen["w"]) + 0.5,
                      y0=xs.index(chosen["x"]) - 0.5, y1=xs.index(chosen["x"]) + 0.5,
                      line=dict(color=INK, width=2))
        with col:
            st.markdown(f"**{title}**")
            show(style(fig, height=300, unified=False))
    st.caption(f"{hp}: median Sharpe over the 9 band (y) and breakout-stop (z) settings in each cell. The outlined cell "
               f"holds the frozen setting ({chosen['Parameters']}). Blue is positive, red negative.")

    # 5. Problems in the strategy as specified
    st.markdown("#### 8. Problems found in the strategy as specified")
    fade_gain = comb["fade_s1"] - comb["net"]
    issues = [
        ("S1 contradicts itself", "S1 is entered because the market is classified as mean-reverting, then it bets "
         f"that a move beyond the band will continue. Fading those breakouts instead changes the all-pair Sharpe "
         f"from {comb['net']:+.2f} to {comb['fade_s1']:+.2f}"
         + (" (better, but still not profitable)." if fade_gain > 0 and comb["fade_s1"] <= 0 else ".")),
        ("The regime test is mostly noise at 1-hour bars", "On these pairs the share of windows that look trending "
         "is close to what a random walk produces, and neither signal predicts the next day's direction (table "
         "above). Classifying regimes from hourly closes does not find real regimes here."),
        ("The S1 stop only catches failures", "A breakout that moves its way soon counts as trending and switches "
         "to the S2 stop, so z decides only the breakouts that never got going, and those are almost all "
         "losers. Tuning z is really choosing how much to lose on a failed breakout."),
        ("Re-entry after a stop was not defined", "Without a rule, a stopped trend trade re-enters on the next bar "
         "while the window still looks trending, paying costs to buy the same move back. This build waits for a "
         "new range to form first; that rule is mine, not the spec's."),
        ("Key measures were not defined", "'Average reversion' and 'average temporary pullback' can be measured many "
         "ways. This build uses mean absolute deviation and the mean dip from the running extreme; other choices "
         "give different results."),
        ("Too many free settings", f"Four settings per pair plus a hidden trend cut-off: {4 * n + 1} numbers fitted "
         f"to {tune_range[0].year}–{tune_range[1].year}. Walk-forward picked "
         f"{int(T['WF parameter sets used'].min())}–{int(T['WF parameter sets used'].max())} different settings "
         "in 6 windows per pair, a sign the 'best' setting is noise."),
        ("Fixed-percent stops ignore volatility", "A z% stop is tight on a quiet pair and loose on a volatile one. "
         "Tuning z per pair only partly fixes this; a stop in units of recent volatility would travel better."),
    ]
    for title, body in issues:
        st.markdown(f"- **{title}.** {body}")
    st.caption("The only pattern that holds across pairs in the sensitivity maps is that longer lookbacks and "
               "wider trend stops lose less, because they trade less. That points toward a slow trend follower, "
               "which is a different strategy.")


# ---- Rebalance frequency
_RAMP_LIGHT = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_RAMP_DARK = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]

with tab_reb:
    st.subheader("How often to check for rebalancing")
    st.write(
        f"The same portfolio, run with the rebalance check every 30 minutes up to every 48 hours, "
        f"{sim_range[0]} to {sim_range[1]}. Strategies, parameters, costs and the objective are unchanged; "
        "exits and stop-losses never wait for a check, so the interval only changes when new signals are "
        "entered and when positions are resized."
    )
    SW = c_sweep(specs, rates, est_range, sim_range, eng, float(capital), float(lev_cap), float(expo_cap), objective,
                 risk_cap_pct / 100.0, floor_pct / 100.0, min_trade_pct / 100.0, cash)
    rrows = []
    for h, r in SW.items():
        hr = r["hourly"]
        m = compute_measures(hr["ret"].to_numpy(), cash_rate=cash, compound=True)
        rrows.append({
            "Every": fmt_hours(h), "hours": h, "Final equity ($)": hr["equity"].iloc[-1],
            "Total return (%)": m["total_return"] * 100, "Sharpe": m["sharpe"],
            "Max drawdown (%)": m["max_drawdown"] * 100, "Rebalances": len(r["rebalances"]),
            "Fills": len(r["fills"]), "Trades": len(r["round_trips"]),
            "Costs paid ($)": r["totals"]["slippage"] + r["totals"]["commission"],
            "Hours over leverage cap": r["hours_over_cap"],
            "Note": f"same checks as {fmt_hours(r['same_as'])}: the data has hourly bars" if r["same_as"] else "",
        })
    rtab = pd.DataFrame(rrows)
    best_r = rtab.loc[rtab["Total return (%)"].idxmax()]
    best_s = rtab.loc[rtab["Sharpe"].idxmax()]
    worst_r = rtab.loc[rtab["Total return (%)"].idxmin()]
    c = st.columns(4)
    c[0].metric("Best total return", best_r["Every"], delta=f"{best_r['Total return (%)']:+.2f}%", delta_color="normal")
    c[1].metric("Best Sharpe", best_s["Every"], delta=f"{best_s['Sharpe']:+.2f}", delta_color="normal")
    c[2].metric("Worst total return", worst_r["Every"], delta=f"{worst_r['Total return (%)']:+.2f}%", delta_color="normal")
    cur = rtab[rtab["hours"] == rebalance_hours]
    if not cur.empty:
        c[3].metric("Your setting", cur["Every"].iloc[0], delta=f"{cur['Total return (%)'].iloc[0]:+.2f}%",
                    delta_color="normal")

    labels = list(rtab["Every"])
    is_cur = (rtab["hours"] == rebalance_hours).to_numpy()
    panels = (("Total return (%)", "%", "Total return"), ("Sharpe", "", "Sharpe ratio"),
              ("Max drawdown (%)", "%", "Max drawdown"), ("Costs paid ($)", "", "Costs paid"))
    for row_panels in (panels[:2], panels[2:]):
        cols = st.columns(2)
        for col, (field, suffix, title) in zip(cols, row_panels):
            fig = go.Figure()
            fig.add_scatter(x=labels, y=rtab[field], mode="lines+markers", name=title,
                            line=dict(color=SLOTS[0], width=2), marker=dict(size=8, color=SLOTS[0]),
                            hovertemplate="%{x}: %{y:,.2f}" + suffix + "<extra></extra>")
            fig.add_scatter(x=[l for l, k in zip(labels, is_cur) if k], y=rtab.loc[is_cur, field], mode="markers",
                            name="Your setting", marker=dict(size=13, color=SLOTS[1], line=dict(color=SURFACE, width=2)),
                            hovertemplate="your setting: %{y:,.2f}" + suffix + "<extra></extra>")
            fig.update_layout(showlegend=False)
            with col:
                st.markdown(f"**{title} by check interval**")
                show(style(fig, height=250, ysuffix=suffix, yprefix="$" if field.startswith("Costs") else "",
                           xtitle="Check for rebalancing every", unified=False).update_xaxes(type="category"))
    st.caption("The larger orange marker is the interval selected in the sidebar.")

    ramp = _RAMP_DARK if DARK else _RAMP_LIGHT
    fig = go.Figure()
    for k_, (h, r) in enumerate(SW.items()):
        if r["same_as"]:
            continue
        eqd = r["hourly"]["equity"].resample("1D").last().dropna()
        fig.add_scatter(x=eqd.index, y=eqd, name=fmt_hours(h), mode="lines",
                        line=dict(color=ramp[k_ % len(ramp)], width=2.5 if h == rebalance_hours else 1.5),
                        hovertemplate="$%{y:,.0f}<extra>every " + fmt_hours(h) + "</extra>")
    fig.add_hline(y=capital, line=dict(color=MUTED, width=1, dash="dot"))
    st.markdown("**Equity curve at each interval** (lighter = more frequent checks)")
    show(style(fig, height=380, ytitle="USD", yprefix="$"))
    st.dataframe(rtab.drop(columns=["hours"]), hide_index=True, width="stretch", column_config={
        "Final equity ($)": st.column_config.NumberColumn(format="%.0f"),
        "Total return (%)": st.column_config.NumberColumn(format="%.2f"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Max drawdown (%)": st.column_config.NumberColumn(format="%.2f"),
        "Costs paid ($)": st.column_config.NumberColumn(format="%.0f"),
    })
    uniq = rtab[rtab["Note"] == ""]
    rho_ret = spearmanr(uniq["hours"], uniq["Total return (%)"]).statistic
    rho_cost = spearmanr(uniq["hours"], uniq["Costs paid ($)"]).statistic
    st.caption(
        f"Reading it: costs fall steadily as checks get less frequent (rank correlation with the interval "
        f"{rho_cost:+.2f}), because fewer resizes are traded. Returns "
        + ("rise with the interval too" if rho_ret > 0.5 else "fall as the interval grows" if rho_ret < -0.5 else
           "do not move steadily with the interval")
        + f" (rank correlation {rho_ret:+.2f}); the spread between best and worst comes largely from which "
        "short-lived signals a schedule happens to catch or miss, so treat a single best interval with caution. "
        "Thirty-minute checks equal hourly ones because the data has one bar per hour."
    )


# ---- Seasonality
with tab_seas:
    st.subheader("Seasonality in each pair")
    st.write(
        "Does a pair's hourly return, or the size of its moves, follow the clock? For each question, 16 "
        "candidate regression models are fitted: no seasonality, hour-of-day and day-of-week dummies, and "
        "daily and weekly Fourier cycles with different numbers of harmonics. The model with the lowest BIC "
        "(Bayesian information criterion, which penalises every extra parameter) is chosen; AIC's choice is "
        f"shown too. Models are fitted on data before {SPLIT_DATE} and checked on the data after it. Daily "
        "returns are also tested with seasonal ARIMA models that allow a weekly (5-day) season."
    )
    SEA = c_seasonality(tuple(available))
    summ = summary_table(SEA)
    nP = len(summ)
    n_none = int((summ["Mean: BIC choice"] == "No seasonality").sum())
    n_vol = int((summ["Volatility: BIC choice"] != "No seasonality").sum())
    n_week = int((summ["Weekly season in daily returns"] == "yes").sum())
    dmean = summ["Mean: best seasonal ΔBIC"]
    dvol = summ["Volatility: ΔBIC vs none"]
    r2v = summ["Volatility: out-of-sample R² (%)"]
    st.info(
        "\n".join([
            f"- **Average returns:** BIC chooses *no seasonality* for {n_none} of {nP} pairs. The best seasonal "
            f"model scores {dmean.min():+.0f} to {dmean.max():+.0f} BIC against it (positive = worse), and its "
            f"out-of-sample R² is about zero ({summ['Mean: out-of-sample R² (%)'].abs().max():.3f}% at most). "
            "There is no hour-of-day or day-of-week effect in returns to trade.",
            f"- **Size of moves:** seasonal in {n_vol} of {nP} pairs, by a wide margin ({dvol.max():,.0f} to "
            f"{dvol.min():,.0f} BIC), explaining {r2v.min():.0f}–{r2v.max():.0f}% of the variation out of sample. "
            "Moves are larger in some hours (the London and New York sessions) and smaller in others. The cost "
            "model already scales slippage with recent volatility, so this is priced in.",
            f"- **Daily returns:** the seasonal ARIMA search picks white noise for "
            f"{int((summ['Daily SARIMA (BIC)'] == '(0,0,0)×(0,0,0)5').sum())} of {nP} pairs; a weekly season is "
            f"chosen for {n_week}.",
        ]), icon="ℹ️")
    st.dataframe(summ, hide_index=True, width="stretch", column_config={
        "Mean: best seasonal ΔBIC": st.column_config.NumberColumn(format="%+.1f"),
        "Mean: HAC p-value": st.column_config.NumberColumn(format="%.3f"),
        "Mean: out-of-sample R² (%)": st.column_config.NumberColumn(format="%.3f"),
        "Volatility: ΔBIC vs none": st.column_config.NumberColumn(format="%+.0f"),
        "Volatility: out-of-sample R² (%)": st.column_config.NumberColumn(format="%.1f"),
        "Ljung-Box p (residuals)": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption("ΔBIC compares a model with the no-seasonality model: below about −10 is strong evidence for the "
               "model, above +10 strong evidence against it. The HAC p-value tests the strongest seasonal "
               "model's terms with standard errors robust to autocorrelation and changing volatility.")

    sp_ = st.selectbox("Pair", list(SEA), key="seas_pair")
    R_ = SEA[sp_]
    mh, vh = R_["mean"]["by_hour"], R_["vol"]["by_hour"]
    left, right = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=mh.index, y=mh["mean"], name="Average return", marker_color=SLOTS[0],
                error_y=dict(type="data", array=2 * mh["se"], color=MUTED, thickness=1),
                hovertemplate="%{x}:00 UTC<br>%{y:.2f} bp<extra></extra>")
    fig.add_scatter(x=mh.index, y=mh["fitted"], name=f"Best seasonal model ({R_['mean']['best_seasonal']})",
                    mode="lines", line=dict(color=SLOTS[1], width=2),
                    hovertemplate="model %{y:.2f} bp<extra></extra>")
    with left:
        st.markdown("**Average hourly return by hour (±2 standard errors)**")
        show(style(fig, height=320, ysuffix=" bp", xtitle="Hour (UTC)", unified=False))
    fig = go.Figure()
    fig.add_bar(x=vh.index, y=np.exp(vh["mean"]) - 1, name="Typical move", marker_color=SLOTS[0],
                hovertemplate="%{x}:00 UTC<br>%{y:.1f} bp<extra></extra>")
    fig.add_scatter(x=vh.index, y=np.exp(vh["fitted"]) - 1, name=f"Chosen model ({R_['vol']['bic_pick']})",
                    mode="lines", line=dict(color=SLOTS[1], width=2),
                    hovertemplate="model %{y:.1f} bp<extra></extra>")
    with right:
        st.markdown("**Typical size of an hourly move by hour**")
        show(style(fig, height=320, ysuffix=" bp", xtitle="Hour (UTC)", unified=False))
    st.caption(f"{sp_}, data before {SPLIT_DATE}. Typical move = geometric average of |return| + 1 bp, minus 1 bp. "
               "On the left the error bars cross zero almost everywhere; on the right the daily cycle is clear.")
    bd = R_["mean"]["by_day"]
    fig = go.Figure()
    fig.add_bar(x=[DAY_NAMES[d] for d in bd.index], y=bd["mean"], marker_color=SLOTS[0],
                error_y=dict(type="data", array=2 * bd["se"], color=MUTED, thickness=1),
                hovertemplate="%{x}: %{y:.2f} bp<extra></extra>")
    st.markdown("**Average hourly return by day of the week (±2 standard errors)**")
    show(style(fig, height=260, ysuffix=" bp", unified=False))
    with st.expander(f"{sp_}: every candidate model, average return"):
        st.dataframe(R_["mean"]["table"], hide_index=True, width="stretch")
    with st.expander(f"{sp_}: every candidate model, size of moves"):
        st.dataframe(R_["vol"]["table"], hide_index=True, width="stretch")
    with st.expander(f"{sp_}: seasonal ARIMA on daily returns"):
        st.dataframe(R_["sarima"]["table"], hide_index=True, width="stretch")
        st.caption(f"Chosen by BIC: {R_['sarima']['bic_pick']}; by AIC: {R_['sarima']['aic_pick']}. Ljung-Box test "
                   f"on the chosen model's residuals (10 lags): p = {R_['sarima']['ljung_box_p']:.3f}.")


# ---- Estimation
with tab_est:
    st.subheader("Per-trade estimates")
    st.write(
        f"Each asset is one strategy on one pair. Its expected return is the mean of its per-trade "
        f"returns over the estimation period ({est_range[0]} to {est_range[1]}) and its risk is their "
        "standard deviation. Returns include carry and are taken before costs, because the optimizer "
        "charges costs itself. The estimates stay fixed during the simulation."
    )
    usable = est[est["n_trades"] > 1].sort_values("std_ret")
    if not usable.empty:
        # A point's label goes above it when it sits above its nearest neighbour, else below,
        # so two close points never share a label position.
        xs = (usable["std_ret"] / usable["std_ret"].abs().max()).to_numpy()
        ys = (usable["mean_ret"] / usable["mean_ret"].abs().max()).to_numpy()
        sides = []
        for j in range(len(usable)):
            d = (xs - xs[j]) ** 2 + (ys - ys[j]) ** 2
            d[j] = np.inf
            near = int(np.argmin(d)) if len(usable) > 1 else j
            sides.append("top center" if ys[j] >= ys[near] else "bottom center")
        fig = go.Figure()
        fig.add_scatter(
            x=usable["std_ret"] * 100, y=usable["mean_ret"] * 100, mode="markers+text",
            text=usable["Asset"], textposition=sides, textfont=dict(color=INK2, size=12),
            marker=dict(color=SLOTS[0], size=11, line=dict(color=SURFACE, width=2)),
            hovertemplate="%{text}<br>risk %{x:.3f}%<br>return %{y:.3f}%<extra></extra>",
        )
        fig.add_hline(y=0, line=dict(color=AXIS, width=1))
        st.markdown("**Expected return against risk, per trade**")
        show(style(fig, height=380, ytitle="Mean return per trade", xtitle="Standard deviation per trade",
                   ysuffix="%", unified=False).update_xaxes(ticksuffix="%"))
    table = est.copy()
    for col in ("mean_ret", "std_ret", "mean_ret_net", "excess_ret", "win_rate"):
        table[col] *= 100
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "n_trades": "Trades (estimation)",
            "mean_ret": st.column_config.NumberColumn("Mean return (%)", format="%.3f"),
            "std_ret": st.column_config.NumberColumn("Std deviation (%)", format="%.3f"),
            "mean_ret_net": st.column_config.NumberColumn("Mean after costs (%)", format="%.3f"),
            "win_rate": st.column_config.NumberColumn("Win rate (%)", format="%.0f"),
            "avg_hours": st.column_config.NumberColumn("Avg hours held", format="%.0f"),
            "excess_ret": st.column_config.NumberColumn("Above cash (%)", format="%.3f"),
            "sim_trades": "Signals in simulation",
        },
    )
    n_pos = int((est["excess_ret"] > 0).sum())
    st.caption(
        f"{n_pos} of {len(est)} assets have a positive expected return after the cash interest given up on "
        "their margin (the Above cash column). The optimizer only ever holds those. Assets with no trades "
        "have a rate gap under the carry minimum."
    )

# ---- Currency risk
with tab_ccy:
    st.subheader("Currency risk")
    st.write(
        "Every position is split into its two currencies: long EUR/USD is long EUR and short USD. "
        "Net exposure per currency is the sum across positions, as a multiple of equity. "
        f"The optimizer keeps each non-USD currency within ±{expo_cap:g}× equity at rebalances."
    )
    expo = sim["exposure"]
    fig = go.Figure()
    for ccy in expo.columns:
        fig.add_scatter(x=expo.index, y=expo[ccy], name=ccy, mode="lines",
                        line=dict(color=CCY_COLOR.get(ccy, MUTED), width=2, shape="hv"),
                        hovertemplate="%{y:+.2f}×<extra>" + ccy + "</extra>")
    st.markdown("**Net exposure per currency (× equity; positive = long)**")
    show(style(fig, ysuffix="×"))
    summary = pd.DataFrame({
        "Currency": expo.columns,
        "Average net exposure (× equity)": expo.mean().to_numpy(),
        "Largest long": expo.max().to_numpy(),
        "Largest short": expo.min().to_numpy(),
        "Hours with a position (%)": (expo.abs() > 1e-9).mean().to_numpy() * 100,
    })
    st.dataframe(summary.round(3), hide_index=True, width="stretch")
    st.markdown("**Converting profits to USD**")
    st.write(
        "Pairs quoted in USD earn their profit in USD. USD/CAD and USD/JPY earn it in Canadian dollars "
        "or yen, which the engine converts to USD at the current rate of the same pair. "
        "A 100,000 USD long on USD/JPY that gains 1% books about 990 USD, not 1,000."
    )

# ---- Data
with tab_data:
    st.subheader("Hourly candles")
    st.write(
        f"{len(available)} pairs, {first_day} to {last_day}. The pairs share "
        f"{len(market.index):,} hourly bars after alignment, about {HOURS_PER_YEAR:,} a year."
    )
    daily = pd.DataFrame({p: frames[p]["close"].resample("1D").last() for p in available}).dropna()
    indexed = daily / daily.iloc[0] * 100.0
    fig = go.Figure()
    for p in available:
        fig.add_scatter(x=indexed.index, y=indexed[p], name=p, mode="lines",
                        line=dict(color=PAIR_COLOR[p], width=2),
                        hovertemplate="%{y:.1f}<extra>" + p + "</extra>")
    st.markdown("**Daily close, indexed to 100 at the first bar**")
    show(style(fig, ytitle="Index (start = 100)"))

    st.markdown("**What one fill costs on each pair**")
    cost_rows = []
    for p in available:
        v = market.pairs[p].vol * 1e4
        cost_rows.append({
            "Pair": p,
            "Hourly volatility, average (bp)": v.mean(),
            "Slippage, average (bp)": v.mean() * slip_mult,
            "Slippage, busiest 5% of hours (bp)": np.quantile(v, 0.95) * slip_mult,
            "Transaction cost (bp)": commission_pct * 100,
            "Round trip, average (bp)": 2 * (v.mean() * slip_mult + commission_pct * 100),
        })
    st.dataframe(pd.DataFrame(cost_rows).round(2), hide_index=True, width="stretch")
    st.caption("1 bp = 0.01% of the traded notional. A round trip is one entry plus one exit.")

    st.markdown("**File checks**")
    st.dataframe(quality_table(raw, frames), hide_index=True, width="stretch")

# ---- Assumptions
with tab_notes:
    st.subheader("How the simulation trades, and what is still a placeholder")
    notes = [
        ("Fills", "Timing", "A signal is decided at a bar's close. Exits fill at the next bar's open. Entries and "
                  "resizes fill at the open of the next rebalance check."),
        ("Fills", "Rebalance check", f"Every {fmt_hours(rebalance_hours)} of clock time from midnight UTC; a check "
                  "that lands while the market is shut happens at the next open. It acts when a signal opened "
                  "or closed since the last check, or leverage drifted above the cap. A signal that opens and "
                  "closes between two checks is never traded."),
        ("Fills", "Trailing stop", f"{trail_pct:g}% behind the best price since entry. The level inside a bar comes from earlier "
                  "bars only. Fills at the stop price, or at the open when a bar gaps past it. If one bar reaches both "
                  "stop and target, the stop fills."),
        ("Fills", "After a stop", "The strategy stays flat until its signal changes."),
        ("Costs", "Transaction cost", f"{commission_pct:g}% of the traded notional on every fill."),
        ("Costs", "Slippage", f"{slip_mult:g} × the pair's standard deviation of hourly returns over the previous "
                  f"{VOL_WINDOW} bars, on every fill including stops."),
        ("Costs", "Netting", "The account holds one net position per pair. Moving weight between two strategies on "
                  "the same pair trades only the net change."),
        ("Costs", "Small resizes", f"A rebalance that would resize a position by under {min_trade_pct:g}% of equity is skipped."),
        ("Account", "Margin", "Margin requirement = 1 ÷ leverage cap. The cap is enforced at rebalances only. "
                    "No margin calls or forced liquidation."),
        ("Account", "Idle equity", "Equity not posted as margin earns the cash rate, accrued continuously."),
        ("Account", "Carry", "Accrued on notional for the time between bars, weekends included. "
                    "Interest rates are constant placeholders, not data."),
        ("Account", "P&L currency", "Profits in CAD or JPY are converted to USD at the current rate."),
        ("Portfolio", "Estimates", "Per trade, before costs, fixed from the estimation period. Trades are treated as independent."),
        ("Portfolio", "Opposite signals on one pair", "Only the side with the higher estimated return may be held."),
        ("Portfolio", "Optimizer risk", "Standard deviation only. Drawdown and CVaR are reported but are not optimizer objectives."),
        ("Portfolio", "Sharpe objective", "Held at the leverage cap, or the currency cap if it binds first. "
                      "The other two objectives weigh trading costs against expected return."),
        ("Portfolio", "Unreachable objective", "Current positions are kept and the rebalance is flagged."),
        ("Tuning", "Method", "Walk-forward and K-fold windows are slices of one full-period backtest per parameter set. "
                   "Fixed grid, fixed notional per trade, pairs pooled at equal notional."),
        ("Not built", "Timeframes", "1-hour only. The 4-hour and daily options are not built."),
        ("Not built", "Bid/ask spread", "Not modeled separately; slippage and the transaction cost stand in for it."),
        ("Not built", "EUR/JPY", "Supported by the code but waiting for its data file."),
    ]
    for strat_ in STRATEGIES.values():   # each block documents its own rules
        notes += [("Strategies", topic, rule) for topic, rule in strat_.notes]
    notes.append(("Pipeline", "Replaceable blocks", "Strategies are blocks in fxsim/strategies/ that register "
                  "themselves; the tuner, estimator, optimizer and rebalance schedule are parameters of the "
                  "stages in fxsim/pipeline.py. Replacing one does not touch the others."))
    st.table(pd.DataFrame(notes, columns=["Area", "Topic", "Rule in this version"]).set_index("Area"))
