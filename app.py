"""Simulated FX Strategy Portfolio: trading simulation page.

Run with:  streamlit run app.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxsim.config import (AFFILIATION, COMMISSION_PCT, FOUNDERS, HOURS_PER_YEAR, PAIRS,
                          PLACEHOLDER_RATES, SLIP_VOL_MULT, TRAIL_STOP_PCT, VOL_WINDOW)
from fxsim.data import align, build_market, load_available, quality_table
from fxsim.engine import run_backtest
from fxsim.estimation import estimate_asset
from fxsim.measures import (MEASURES, PERCENT_MEASURES, compute_measures, drawdown_series,
                            longest_underwater, trade_stats)
from fxsim.optimizer import OBJECTIVES
from fxsim.portfolio import Asset, simulate
from fxsim.strategies import CARRY_MIN_DIFF, STRATEGIES, carry_diff, make_signal
from fxsim.tuning import kfold, returns_matrix, suggest_from_kfold, walk_forward

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
    R = returns_matrix(mk, frames, key, list(pairs), dict(rates), i0, i1, **_engine_kwargs(eng))
    wf, oos = walk_forward(R, measure, train_m, test_m, cash)
    folds, params = kfold(R, measure, k, embargo, cash)
    return {"labels": list(R.columns), "wf": wf, "oos": oos, "folds": folds, "params": params,
            "suggested": suggest_from_kfold(params, measure)}


@st.cache_data(show_spinner=False)
def c_backtest(key, pair, params, rates, start, end, eng):
    _, frames, _, mk = get_market()
    i0, i1 = mk.window(start, end)
    rates = dict(rates)
    sig = make_signal(key, frames[pair], pair, params, rates)
    res = run_backtest(mk, pair, sig, i0, i1, carry_diff_pct=carry_diff(pair, rates), **_engine_kwargs(eng))
    return res.trades, res.hourly


@st.cache_data(show_spinner="Simulating the portfolio")
def c_simulate(specs, rates, est_range, sim_range, eng, capital, lev_cap, expo_cap, objective,
               risk_cap, reward_floor, min_trade, cash):
    _, _, _, mk = get_market()
    assets, rows = [], []
    for key, pair, params in specs:
        est_trades, _ = c_backtest(key, pair, params, rates, *est_range, eng)
        sim_trades, _ = c_backtest(key, pair, params, rates, *sim_range, eng)
        est = estimate_asset(est_trades, cash, 1.0 / lev_cap)
        name = f"{STRATEGIES[key].short} · {pair}"
        assets.append(Asset(name, key, pair, sim_trades, est["excess_ret"], est["std_ret"]))
        rows.append({"Asset": name, **est, "sim_trades": len(sim_trades)})
    i0, i1 = mk.window(*sim_range)
    out = simulate(mk, assets, i0, i1, capital=capital, leverage_cap=lev_cap, expo_cap=expo_cap,
                   objective=objective, risk_cap=risk_cap, reward_floor=reward_floor,
                   commission_pct=eng[2], slip_vol_mult=eng[3], min_trade_frac=min_trade,
                   cash_rate=cash, rates=dict(rates))
    out["estimates"] = pd.DataFrame(rows)
    return out


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
    with about_box:
        st.header("About")
        st.write(
            "A multi-strategy currency portfolio. Trend-following and carry-trade signals run on the "
            "major FX pairs at the 1-hour timeframe. Every open trade is treated as a portfolio asset, "
            "and a leveraged margin account decides how much of each to hold."
        )
        pair_names = ", ".join("/".join(PAIRS[p]) for p in sel_pairs) or "none selected"
        signal_names = ", ".join(STRATEGIES[k].name for k in sel_strategies) or "none selected"
        stop_text = f"{trail_pct:g}% trailing stop on every trade" if trail_pct > 0 else "no stop-loss"
        st.markdown(
            f"- **Market:** {pair_names}\n"
            f"- **Signals:** {signal_names}\n"
            f"- **Sizing:** weights re-optimized at every entry and exit, leverage capped at {lev_cap:g}×\n"
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
        st.divider()

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

tab_over, tab_port, tab_bt, tab_tune, tab_est, tab_ccy, tab_data, tab_notes = st.tabs(
    ["Overview", "Portfolio", "Strategy trades", "Tuning", "Estimation", "Currency risk", "Data", "Assumptions"]
)

# ---- Tuning (computed first: everything downstream uses the frozen parameters)
frozen: dict[str, tuple] = {}
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
        t = c_tune(key, tuple(sel_pairs), rates, *tune_range, eng, tune_measure,
                   int(train_months), int(test_months), int(n_folds), int(embargo), cash)
        st.markdown(f"#### {strat.name}")
        st.caption(strat.description)

        choice = st.selectbox(
            "Frozen parameter set (default: best mean test-fold score in K-fold)",
            t["labels"], index=t["labels"].index(t["suggested"]), key=f"frozen_{key}_{t['suggested']}",
        )
        frozen[key] = strat.grid[t["labels"].index(choice)]

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
specs = tuple((key, pair, frozen[key]) for key in sel_strategies for pair in sel_pairs)
sim = c_simulate(specs, rates, est_range, sim_range, eng, float(capital), float(lev_cap), float(expo_cap),
                 objective, risk_cap_pct / 100.0, floor_pct / 100.0, min_trade_pct / 100.0, cash)
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
        f"Objective: **{OBJECTIVES[objective]}**. A signal decided at a bar's close trades at the next "
        "bar's open. The account re-optimizes over the open signals at every entry and exit, and holds "
        "one net position per pair."
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

# ---- Strategy trades
with tab_bt:
    st.subheader("Strategy trades")
    c = st.columns(2)
    bt_key = c[0].selectbox("Strategy", sel_strategies, format_func=lambda k: STRATEGIES[k].name)
    bt_period = c[1].selectbox("Period", list(PERIODS), index=2)
    strat = STRATEGIES[bt_key]
    st.caption(
        f"{strat.description} Frozen parameters: {strat.label(frozen[bt_key])}. Every trade is "
        f"{BACKTEST_NOTIONAL:,.0f} USD notional, with a {trail_pct:g}% trailing stop, "
        f"{commission_pct:g}% transaction cost per fill and volatility-scaled slippage."
    )
    runs = {p: c_backtest(bt_key, p, frozen[bt_key], rates, *PERIODS[bt_period], eng) for p in sel_pairs}

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
        st.info("No trades in this period. For the carry trade this means the rate gap is under "
                f"{CARRY_MIN_DIFF} percentage points.")
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
        ("Fills", "Timing", "A signal decided at a bar's close fills at the next bar's open. Rebalance trades fill at the same open."),
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
        ("Strategies", "Carry trade", "The trend filter that splits carry into separate trades is a placeholder rule."),
        ("Not built", "Timeframes", "1-hour only. The 4-hour and daily options are not built."),
        ("Not built", "Bid/ask spread", "Not modeled separately; slippage and the transaction cost stand in for it."),
        ("Not built", "EUR/JPY", "Supported by the code but waiting for its data file."),
    ]
    st.table(pd.DataFrame(notes, columns=["Area", "Topic", "Rule in this version"]).set_index("Area"))
