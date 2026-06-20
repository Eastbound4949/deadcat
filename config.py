"""
config.py — Momentum Bot. Set via env vars (Railway) or edit directly (local).
Binance UM Futures blocked on US IPs — use bybit or gate instead.
"""

import os

# ─── Exchange ──────────────────────────────────────────────────────────────────
# binanceusdm = Binance USDM Futures (non-US only)
# bybit = Bybit (works from US Railway servers)
# gate  = Gate.io futures (works from US)
EXCHANGE   = os.environ.get("EXCHANGE",   "binanceusdm")
API_KEY    = os.environ.get("API_KEY",    "")
API_SECRET = os.environ.get("API_SECRET", "")

# ─── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ─── Pairs to scan (futures perps with good liquidity) ────────────────────────
# Format: BASE/QUOTE:SETTLE (ccxt perpetual format)
# Binance UM: BTC/USDT:USDT  |  Bybit: BTC/USDT:USDT  |  Gate: BTC/USDT:USDT
_default_pairs = ",".join([
    "SOL/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "INJ/USDT:USDT",
    "SUI/USDT:USDT", "APT/USDT:USDT",  "ARB/USDT:USDT",  "OP/USDT:USDT",
    "TIA/USDT:USDT", "SEI/USDT:USDT",  "RUNE/USDT:USDT", "AAVE/USDT:USDT",
    "LDO/USDT:USDT", "DYDX/USDT:USDT", "WLD/USDT:USDT",  "CRV/USDT:USDT",
    "GRT/USDT:USDT", "FTM/USDT:USDT",  "NEAR/USDT:USDT", "ATOM/USDT:USDT",
])
SYMBOLS = os.environ.get("SYMBOLS", _default_pairs).split(",")

# ─── Signal thresholds ────────────────────────────────────────────────────────
VOLUME_SPIKE_MIN     = float(os.environ.get("VOLUME_SPIKE_MIN",    "4.0"))  # × rolling avg
PRICE_ROC_MIN        = float(os.environ.get("PRICE_ROC_MIN",       "3.0"))  # % in 2 bars (min)
PRICE_ROC_MAX        = float(os.environ.get("PRICE_ROC_MAX",       "9.0"))  # % in 2 bars (max — skip if already played out)
VOLUME_LOOKBACK_BARS = int(os.environ.get("VOLUME_LOOKBACK_BARS",  "20"))   # bars for rolling avg baseline

# ─── Risk & position sizing ───────────────────────────────────────────────────
PAPER_STARTING_BALANCE = float(os.environ.get("PAPER_STARTING_BALANCE", "10000"))
RISK_PER_TRADE         = float(os.environ.get("RISK_PER_TRADE",         "0.015"))  # 1.5% of balance
MAX_POSITION_PCT       = float(os.environ.get("MAX_POSITION_PCT",       "0.50"))   # cap margin at 50% of balance
LEVERAGE               = int(os.environ.get("LEVERAGE",                  "7"))
SL_PCT                 = float(os.environ.get("SL_PCT",                  "0.025"))  # 2.5% hard stop
TRAIL_PCT              = float(os.environ.get("TRAIL_PCT",               "0.018"))  # 1.8% trailing stop off peak
MIN_RR                 = float(os.environ.get("MIN_RR",                  "1.5"))    # min R:R to take entry
MAX_HOLD_MINUTES       = int(os.environ.get("MAX_HOLD_MINUTES",          "15"))     # time-based exit if no follow-through

# ─── Session & scheduler ─────────────────────────────────────────────────────
SESSION_START_UTC     = int(os.environ.get("SESSION_START_UTC",     "6"))   # 06:00 UTC
SESSION_END_UTC       = int(os.environ.get("SESSION_END_UTC",       "22"))  # 22:00 UTC
SCAN_INTERVAL_SECONDS = int(os.environ.get("SCAN_INTERVAL_SECONDS", "30"))

# ─── Mode ─────────────────────────────────────────────────────────────────────
# DRY_RUN=true → Telegram alerts only, no paper trade logged, no real orders
# DRY_RUN=false → paper trades logged to CSV (and live orders if LIVE=true)
DRY_RUN  = os.environ.get("DRY_RUN",  "true").lower()  == "true"
LIVE     = os.environ.get("LIVE",     "false").lower() == "true"  # real exchange orders
LOG_FILE = os.environ.get("LOG_FILE", "trades_log.csv")
