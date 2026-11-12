"""Project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "currencydata_raw"

# About panel. Edit the founder lines here; `about` is a one-line description
# shown under each name (role, background, focus) and may be left empty.
FOUNDERS = [
    {"name": "Austin DeCicco", "about": ""},
    {"name": "Neel Barve", "about": ""},
]
AFFILIATION = "Rutgers University · Algorithmic Trading and Portfolio Management · Fall 2026"

# pair -> (base currency, quote currency).
# EURJPY is part of the portfolio but its file is not in the folder yet;
# it is picked up automatically once a file named EURJPY_1h_*.csv is added.
PAIRS = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "AUDUSD": ("AUD", "USD"),
    "USDCAD": ("USD", "CAD"),
    "USDJPY": ("USD", "JPY"),
    "EURJPY": ("EUR", "JPY"),
}

CURRENCIES = ["USD", "EUR", "GBP", "AUD", "CAD", "JPY"]

# 24 hours x 5 days x 52 weeks. The data has about 6,236 hourly bars a year.
HOURS_PER_YEAR = 6240
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Execution defaults.
COMMISSION_PCT = 0.01    # transaction cost per fill, percent of traded notional
TRAIL_STOP_PCT = 2.5     # trailing stop-loss, percent from the best price since entry
SLIP_VOL_MULT = 0.05     # slippage per fill = this x the pair's hourly volatility
VOL_WINDOW = 24          # bars of hourly returns behind the volatility estimate

# PLACEHOLDER interest rates, percent a year, one constant per currency.
# They approximate mid-2025 policy rates and are NOT from the dataset: the
# hourly candle files hold no interest-rate data. Replace with a historical
# series before trusting any carry number.
PLACEHOLDER_RATES = {
    "USD": 4.33,
    "EUR": 2.00,
    "GBP": 4.25,
    "AUD": 3.85,
    "CAD": 2.75,
    "JPY": 0.50,
}
