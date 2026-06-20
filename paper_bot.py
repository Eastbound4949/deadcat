"""
paper_bot.py — DCB Dynamic Scanner Paper Bot

Scans ALL Binance UM Futures tickers every 4H for tokens with >20% daily gain.
Adds new pumpers to watchlist. Monitors watchlist for Dead Cat Bounce short signal.
One paper short per pair, then removes it from watchlist.

Signal (same as backtest_dcb.py v2):
  Post-pump filter: price < PUMP_DROP_PCT × 90d high AND below EMA50
  Bounce confirm : last CONFIRM_BARS bars below EMA20
  Volume filter  : bounce vol < VOL_RATIO × prior dump vol
  Rejection candle: high >= EMA20, close < EMA20, close < open

Exit:
  SL  : max(candle_high, EMA20) + ATR14 × ATR_MULT  (fixed, no trail)
  TP1 : entry × (1 - TP1_PCT)  — close 60% of position
  TP2 : entry × (1 - TP2_PCT)  — close remaining 40%
  TIME: MAX_HOLD_BARS × 4H bars (15 days)

State files (in DATA_DIR or script dir):
  watchlist.json   — pairs being monitored
  positions.json   — open paper positions
  dcb_paper_trades.csv — completed trade log

Deploy: EXCHANGE=binanceusdm (EU Railway region; or bybit for US).
"""

import sys, os, json, csv, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests
from apscheduler.schedulers.blocking import BlockingScheduler

sys.stdout.reconfigure(encoding="utf-8")

_DIR = Path(os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__))))

# ─── Config ──────────────────────────────────────────────────────────────────
EXCHANGE      = os.environ.get("EXCHANGE",      "binanceusdm")
API_KEY       = os.environ.get("API_KEY",       "")
API_SECRET    = os.environ.get("API_SECRET",    "")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

PUMP_MIN_PCT   = float(os.environ.get("PUMP_MIN_PCT",   "20.0"))  # 24h gain to trigger
WATCHLIST_DAYS = int(os.environ.get("WATCHLIST_DAYS",   "30"))    # expire after N days
PAPER_BALANCE  = float(os.environ.get("PAPER_BALANCE",  "10000"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.015"))

# DCB signal params
EMA_FAST      = int(os.environ.get("EMA_FAST",      "20"))
EMA_SLOW      = int(os.environ.get("EMA_SLOW",      "50"))
CONFIRM_BARS  = int(os.environ.get("CONFIRM_BARS",  "5"))
PUMP_LOOKBACK = int(os.environ.get("PUMP_LOOKBACK", "540"))  # 90d on 4H
PUMP_DROP_PCT = float(os.environ.get("PUMP_DROP_PCT","0.50"))
ATR_PERIOD    = int(os.environ.get("ATR_PERIOD",    "14"))
ATR_MULT      = float(os.environ.get("ATR_MULT",    "1.5"))
VOL_RATIO     = float(os.environ.get("VOL_RATIO",   "0.80"))
VOL_LOOKBACK  = int(os.environ.get("VOL_LOOKBACK",  "10"))
TP1_PCT       = float(os.environ.get("TP1_PCT",     "0.20"))
TP2_PCT       = float(os.environ.get("TP2_PCT",     "0.40"))
MAX_HOLD_BARS = int(os.environ.get("MAX_HOLD_BARS", "90"))   # 15 days on 4H


# ─── Telegram ────────────────────────────────────────────────────────────────

def tg(msg: str):
    if not TELEGRAM_TOKEN:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"[tg] {e}")


# ─── Exchange ─────────────────────────────────────────────────────────────────

_ex: ccxt.Exchange | None = None

def ex() -> ccxt.Exchange:
    global _ex
    if _ex is None:
        # Gate.io perpetuals use 'swap' type; Binance/Bybit use 'future'
        market_type = "swap" if EXCHANGE in ("gate", "gateio") else "future"
        _ex = getattr(ccxt, EXCHANGE)({"apiKey": API_KEY, "secret": API_SECRET,
                                        "options": {"defaultType": market_type}})
    return _ex


# ─── State helpers ────────────────────────────────────────────────────────────

def load_watchlist() -> dict:
    f = _DIR / "watchlist.json"
    return json.loads(f.read_text()) if f.exists() else {}

def save_watchlist(wl: dict):
    (_DIR / "watchlist.json").write_text(json.dumps(wl, indent=2))

def load_positions() -> dict:
    f = _DIR / "positions.json"
    return json.loads(f.read_text()) if f.exists() else {}

def save_positions(pos: dict):
    (_DIR / "positions.json").write_text(json.dumps(pos, indent=2))

def load_balance() -> float:
    f = _DIR / "balance.json"
    return json.loads(f.read_text())["balance"] if f.exists() else PAPER_BALANCE

def save_balance(b: float):
    (_DIR / "balance.json").write_text(json.dumps({"balance": round(b, 2)}))

def log_trade(row: dict):
    f = _DIR / "dcb_paper_trades.csv"
    exists = f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


# ─── Scanner ─────────────────────────────────────────────────────────────────

def scan_movers(wl: dict) -> list[str]:
    """
    Scan all USDT perpetuals on the configured exchange for >PUMP_MIN_PCT gainers.
    Gate.io (swap) and KuCoin both work from Railway US — Binance/Bybit are blocked.
    Gate.io has 700+ perps incl micro-caps where pumps are most common.
    """
    print("[scanner] Fetching all tickers ...", flush=True)
    market_type = "swap" if EXCHANGE in ("gate", "gateio") else "future"
    try:
        scan_ex = getattr(ccxt, EXCHANGE)({"options": {"defaultType": market_type}})
        raw = scan_ex.fetch_tickers()
    except Exception as e:
        print(f"[scanner] fetch_tickers failed: {e}")
        return []

    perps = {s: t for s, t in raw.items() if "/USDT:USDT" in s}
    print(f"[scanner] {len(perps)} USDT perps fetched")

    new_syms = []
    for sym, t in perps.items():
        pct   = float(t.get("percentage", 0) or 0)
        price = float(t.get("last", 0) or 0)
        if price < 0.0001:
            continue
        if pct >= PUMP_MIN_PCT and sym not in wl:
            wl[sym] = {
                "added_utc": datetime.now(timezone.utc).isoformat(),
                "pump_pct":  round(pct, 1),
                "traded":    False,
            }
            new_syms.append(sym)
            print(f"  [+] {sym:30s} +{pct:.1f}% daily pump")

    return new_syms


def prune_watchlist(wl: dict, pos: dict) -> dict:
    """Remove expired or traded pairs (keep if position still open)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=WATCHLIST_DAYS)
    return {
        sym: info for sym, info in wl.items()
        if sym in pos                                         # keep if position open
        or (not info["traded"]                                # keep if not yet traded
            and datetime.fromisoformat(info["added_utc"]) > cutoff)
    }


# ─── OHLCV + indicators ──────────────────────────────────────────────────────

def fetch_4h(sym: str, bars: int = 600) -> pd.DataFrame | None:
    try:
        raw = ex().fetch_ohlcv(sym, "4h", limit=bars)
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df["ema_fast"]    = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
        df["ema_slow"]    = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
        df["recent_high"] = df["high"].rolling(PUMP_LOOKBACK, min_periods=1).max()
        prev_close = df["close"].shift(1)
        tr = pd.concat([(df["high"]-df["low"]),
                        (df["high"]-prev_close).abs(),
                        (df["low"] -prev_close).abs()], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
        return df
    except Exception as e:
        print(f"  [ohlcv] {sym}: {e}")
        return None


def check_signal(df: pd.DataFrame) -> dict | None:
    """Return signal dict if DCB short entry is valid on latest bar, else None."""
    i = len(df) - 1
    min_i = max(EMA_SLOW, PUMP_LOOKBACK, ATR_PERIOD) + CONFIRM_BARS + 5
    if i < min_i:
        return None

    row       = df.iloc[i]
    close     = float(row["close"])
    high      = float(row["high"])
    open_     = float(row["open"])
    ema_fast  = float(row["ema_fast"])
    ema_slow  = float(row["ema_slow"])
    rec_high  = float(row["recent_high"])
    atr       = float(row["atr"])

    if round(close, 4) == 0 or atr <= 0:
        return None

    # 1. Post-pump filter
    if not (close < rec_high * PUMP_DROP_PCT and close < ema_slow):
        return None

    # 2. Prior bars all below EMA_FAST
    prev_c = df["close"].iloc[i - CONFIRM_BARS: i].values
    prev_e = df["ema_fast"].iloc[i - CONFIRM_BARS: i].values
    if not (prev_c < prev_e).all():
        return None

    # 3. Volume filter
    bounce_vol = float(df["volume"].iloc[i - CONFIRM_BARS: i + 1].mean())
    prior_start = max(0, i - CONFIRM_BARS - VOL_LOOKBACK)
    prior_vol   = float(df["volume"].iloc[prior_start: i - CONFIRM_BARS].mean())
    if prior_vol > 0 and bounce_vol >= prior_vol * VOL_RATIO:
        return None

    # 4. Rejection candle
    if not (high >= ema_fast * 0.995 and close < ema_fast and close < open_):
        return None

    sl = max(high, ema_fast) + ATR_MULT * atr
    rr1 = (close - close * (1 - TP1_PCT)) / (sl - close)
    if rr1 < 0.8:
        return None

    return {
        "entry":    close,
        "sl":       sl,
        "tp1":      close * (1 - TP1_PCT),
        "tp2":      close * (1 - TP2_PCT),
        "atr":      atr,
        "sl_risk":  (sl - close) / close,
    }


# ─── Position management ─────────────────────────────────────────────────────

def open_position(sym: str, sig: dict, balance: float) -> dict:
    risk_usdt = balance * RISK_PER_TRADE
    sl_dist   = sig["sl"] - sig["entry"]
    notional  = risk_usdt / (sl_dist / sig["entry"])
    notional  = min(notional, balance * 0.50)

    return {
        "symbol":      sym,
        "entry":       sig["entry"],
        "sl":          sig["sl"],
        "tp1":         sig["tp1"],
        "tp2":         sig["tp2"],
        "sl_risk":     sig["sl_risk"],
        "notional":    notional,
        "margin":      notional * sig["sl_risk"] * (1 / RISK_PER_TRADE) * RISK_PER_TRADE,
        "tp1_hit":     False,
        "entry_utc":   datetime.now(timezone.utc).isoformat(),
        "bar_count":   0,
    }


def check_exits(pos: dict, df: pd.DataFrame) -> tuple[str | None, float]:
    """Check last bar for SL/TP hits. Returns (reason, exit_price) or (None, 0)."""
    bar  = df.iloc[-1]
    hi   = float(bar["high"])
    lo   = float(bar["low"])
    cls  = float(bar["close"])

    pos["bar_count"] = pos.get("bar_count", 0) + 1

    # SL: short = price goes UP above SL
    if hi >= pos["sl"]:
        return "SL", pos["sl"]

    # TP1
    if not pos["tp1_hit"] and lo <= pos["tp1"]:
        pos["tp1_hit"] = True
        # continue monitoring for TP2

    # TP2
    if pos["tp1_hit"] and lo <= pos["tp2"]:
        return "TP2", pos["tp2"]

    # TIME exit
    if pos["bar_count"] >= MAX_HOLD_BARS:
        return "TP1+TIME" if pos["tp1_hit"] else "TIME", cls

    return None, 0.0


def close_position(pos: dict, reason: str, exit_price: float, balance: float) -> tuple[float, float]:
    """Returns (pnl_usdt, new_balance)."""
    entry = pos["entry"]
    notional = pos["notional"]

    if pos.get("tp1_hit") and reason not in ("SL", "TIME"):
        # Blended: 60% exited at TP1, 40% at current exit
        pnl = 0.60 * (entry - pos["tp1"]) / entry * notional + \
              0.40 * (entry - exit_price) / entry * notional
    else:
        pnl = (entry - exit_price) / entry * notional

    new_bal = max(balance + pnl, 0.01)
    return round(pnl, 2), round(new_bal, 2)


# ─── Main loop ────────────────────────────────────────────────────────────────

def run():
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*56}\n[bot] {now_str}")

    wl      = load_watchlist()
    pos     = load_positions()
    balance = load_balance()

    trades_this_run = 0
    exits_this_run  = 0

    # ── 1. Check open positions ───────────────────────────────────────────────
    for sym in list(pos.keys()):
        df = fetch_4h(sym, bars=PUMP_LOOKBACK + 50)
        if df is None or len(df) < 5:
            continue

        reason, exit_price = check_exits(pos[sym], df)
        if reason:
            pnl, balance = close_position(pos[sym], reason, exit_price, balance)
            save_balance(balance)

            msg = (f"*EXIT {reason}* — {sym}\n"
                   f"Entry: ${pos[sym]['entry']:.4f} -> Exit: ${exit_price:.4f}\n"
                   f"P&L: ${pnl:+,.2f} | Balance: ${balance:,.2f}")
            print(f"[exit] {sym} {reason} P&L ${pnl:+,.2f} bal ${balance:,.2f}")
            tg(msg)

            log_trade({
                "utc":        now_str,
                "symbol":     sym,
                "reason":     reason,
                "entry":      pos[sym]["entry"],
                "exit":       exit_price,
                "pnl_usd":    pnl,
                "balance":    balance,
                "bars_held":  pos[sym].get("bar_count", 0),
                "pump_pct":   wl.get(sym, {}).get("pump_pct", "?"),
            })

            del pos[sym]
            save_positions(pos)
            exits_this_run += 1
        else:
            upnl = (pos[sym]["entry"] - float(df["close"].iloc[-1])) / pos[sym]["entry"] * pos[sym]["notional"]
            print(f"[hold] {sym:30s} bars={pos[sym].get('bar_count',0):3d} uPnL=${upnl:+,.2f}")

    # ── 2. Scan for new pumpers ───────────────────────────────────────────────
    new_syms = scan_movers(wl)
    if new_syms:
        wl = prune_watchlist(wl, pos)
        save_watchlist(wl)
        msg = f"*{len(new_syms)} new pumpers added to watchlist*\n" + \
              "\n".join(f"  {s.replace('/USDT:USDT','')} +{wl[s]['pump_pct']}%" for s in new_syms[:10])
        tg(msg)

    # ── 3. Check watchlist for DCB signal ────────────────────────────────────
    pending = {s: info for s, info in wl.items()
               if not info["traded"] and s not in pos}

    print(f"[bot] Watchlist: {len(wl)} pairs | Pending: {len(pending)} | Open: {len(pos)}")

    for sym, info in pending.items():
        if sym in pos:
            continue  # already have a position

        df = fetch_4h(sym, bars=PUMP_LOOKBACK + 50)
        if df is None or len(df) < max(EMA_SLOW, PUMP_LOOKBACK) + 20:
            continue
        time.sleep(0.1)  # rate limit

        sig = check_signal(df)
        if sig is None:
            continue

        # Open position
        p = open_position(sym, sig, balance)
        pos[sym] = p
        save_positions(pos)

        # Mark as traded in watchlist
        wl[sym]["traded"] = True
        save_watchlist(wl)

        msg = (f"*DCB SHORT SIGNAL — {sym}*\n"
               f"Pump was: +{info['pump_pct']}% (added {info['added_utc'][:10]})\n"
               f"Entry: ${sig['entry']:.4f}\n"
               f"SL: ${sig['sl']:.4f} ({sig['sl_risk']*100:.1f}% risk)\n"
               f"TP1: ${sig['tp1']:.4f} (-{TP1_PCT*100:.0f}%)\n"
               f"TP2: ${sig['tp2']:.4f} (-{TP2_PCT*100:.0f}%)\n"
               f"Notional: ${p['notional']:,.2f} | Balance: ${balance:,.2f}")
        print(f"[entry] {sym} entry=${sig['entry']:.4f} SL=${sig['sl']:.4f} TP2=${sig['tp2']:.4f}")
        tg(msg)
        trades_this_run += 1

    # ── 4. Status summary ─────────────────────────────────────────────────────
    print(f"[bot] Done | Balance: ${balance:,.2f} | Open: {len(pos)} | "
          f"New trades: {trades_this_run} | Exits: {exits_this_run}")


# ─── Startup + scheduler ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print(" DCB Dynamic Scanner Paper Bot")
    print(f" Exchange : {EXCHANGE}")
    print(f" Pump gate: +{PUMP_MIN_PCT}% daily → watchlist")
    print(f" Signal   : EMA{EMA_FAST} rejection + vol<{VOL_RATIO:.0%} + ATR{ATR_PERIOD}x{ATR_MULT} SL")
    print(f" Exit     : TP1 -{TP1_PCT:.0%} (60%) | TP2 -{TP2_PCT:.0%} (40%) | {MAX_HOLD_BARS}bar max")
    print(f" Balance  : ${load_balance():,.2f} (paper)")
    print("=" * 56)

    tg(f"*DCB Scanner Bot started*\nExchange: {EXCHANGE}\nGate: +{PUMP_MIN_PCT}% daily\nBalance: ${load_balance():,.2f}")

    run()

    scheduler = BlockingScheduler()
    scheduler.add_job(run, "interval", hours=4)
    print("\n[bot] Scanning every 4H. Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[bot] Stopped.")
        tg("DCB Scanner Bot stopped.")
