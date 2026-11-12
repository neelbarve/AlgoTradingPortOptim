"""Load the hourly candle files and put the pairs on one shared time grid."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, PAIRS, SECONDS_PER_YEAR, VOL_WINDOW

COLUMNS = ["open", "high", "low", "close", "volume"]


def load_pair(pair: str, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    """Read one pair's 1-hour candles. Returns None when the file is missing."""
    files = sorted(Path(data_dir).glob(f"{pair}_1h_*.csv"))
    if not files:
        return None
    df = pd.read_csv(files[-1])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[COLUMNS].astype(float)


def load_available(data_dir: Path = DATA_DIR) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load every configured pair that has a file. Returns (frames, missing pairs)."""
    frames, missing = {}, []
    for pair in PAIRS:
        df = load_pair(pair, data_dir)
        if df is None:
            missing.append(pair)
        else:
            frames[pair] = df
    return frames, missing


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Keep only the hours every pair has, so all pairs share one time grid."""
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    return {pair: df.loc[common] for pair, df in frames.items()}


def quality_table(raw: dict[str, pd.DataFrame], aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per pair: size, range and basic integrity checks."""
    rows = []
    for pair, df in raw.items():
        gaps = df.index.to_series().diff().dt.total_seconds().div(3600)
        bad_range = (df["high"] < df[["open", "close"]].max(axis=1)) | (
            df["low"] > df[["open", "close"]].min(axis=1)
        )
        rows.append(
            {
                "Pair": pair,
                "Bars in file": len(df),
                "First bar (UTC)": df.index.min().strftime("%Y-%m-%d %H:%M"),
                "Last bar (UTC)": df.index.max().strftime("%Y-%m-%d %H:%M"),
                "Missing values": int(df.isna().sum().sum()),
                "High/low errors": int(bad_range.sum()),
                "Gaps inside the week": int(((gaps > 1) & (gaps < 48)).sum()),
                "Bars dropped to align": len(df) - len(aligned[pair]),
            }
        )
    return pd.DataFrame(rows)


def quote_to_usd(pair: str, frames: dict[str, pd.DataFrame]) -> np.ndarray:
    """USD value of one unit of the pair's quote currency at each bar's close."""
    quote = PAIRS[pair][1]
    n = len(frames[pair])
    if quote == "USD":
        return np.ones(n)
    if f"USD{quote}" in frames:
        return 1.0 / frames[f"USD{quote}"]["close"].to_numpy()
    if f"{quote}USD" in frames:
        return frames[f"{quote}USD"]["close"].to_numpy()
    raise ValueError(f"No pair available to convert {quote} to USD for {pair}")


@dataclass
class PairArrays:
    """Plain arrays for the engine loops."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    q_usd: np.ndarray  # USD per one unit of the quote currency
    vol: np.ndarray    # std of hourly returns over the trailing window, known before the bar opens


@dataclass
class Market:
    index: pd.DatetimeIndex
    dt_years: np.ndarray  # time since the previous bar, in years (covers weekends)
    pairs: dict[str, PairArrays]

    def window(self, start, end) -> tuple[int, int]:
        """Bar positions [i0, i1) covering the dates start..end inclusive."""
        t0 = pd.Timestamp(start, tz="UTC")
        t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        return int(self.index.searchsorted(t0)), int(self.index.searchsorted(t1))


def trailing_vol(close: pd.Series, window: int = VOL_WINDOW) -> np.ndarray:
    """Std of hourly returns over the last `window` bars, lagged one bar.

    The value at bar i uses closes up to bar i-1 only, so it is known when
    bar i opens. Weekend gaps count as one return, which makes the first
    hours of the week more expensive to trade, as they are in practice.
    """
    vol = close.pct_change().rolling(window).std().shift(1)
    return vol.bfill().to_numpy()


def build_market(aligned: dict[str, pd.DataFrame]) -> Market:
    index = next(iter(aligned.values())).index
    dt = index.to_series().diff().dt.total_seconds().fillna(0.0).to_numpy() / SECONDS_PER_YEAR
    pairs = {
        pair: PairArrays(
            open=df["open"].to_numpy(),
            high=df["high"].to_numpy(),
            low=df["low"].to_numpy(),
            close=df["close"].to_numpy(),
            q_usd=quote_to_usd(pair, aligned),
            vol=trailing_vol(df["close"]),
        )
        for pair, df in aligned.items()
    }
    return Market(index=index, dt_years=dt, pairs=pairs)
