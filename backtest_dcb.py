"""
backtest_dcb.py — Dead Cat Bounce SHORT strategy backtest (v2: vol filter + ATR SL).

Signal:
  1. Post-pump filter : price < 50% of 90-day high AND below EMA50
  2. DCB bounce       : last CONFIRM_BARS closes below EMA20
  3. Volume filter    : bounce volume < VOL_RATIO × prior period volume (weak buying)
  4. Rejection candle : high >= EMA20, close < EMA20, close < open
  5. Short at close of rejection candle

Exit:
  SL  : max(candle_high, EMA20) + ATR_MULT × ATR14  (volatility-adaptive)
  TP1 : entry × (1 - TP1_PCT)  — close 60%
  TP2 : entry × (1 - TP2_PCT)  — close 40% (runner)
  TIME: MAX_HOLD_BARS bars

Tests 10 dead post-pump tokens on Binance UM Futures (4H data).
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ─── Parameters ──────────────────────────────────────────────────────────────
EXCHANGE      = os.environ.get("EXCHANGE",      "binanceusdm")
API_KEY       = os.environ.get("API_KEY",       "")
API_SECRET    = os.environ.get("API_SECRET",    "")
BACKTEST_DAYS = int(os.environ.get("BACKTEST_DAYS", "365"))

EMA_FAST      = int(os.environ.get("EMA_FAST",    "20"))
EMA_SLOW      = int(os.environ.get("EMA_SLOW",    "50"))
CONFIRM_BARS  = int(os.environ.get("CONFIRM_BARS", "5"))   # bars below EMA before bounce
PUMP_LOOKBACK = int(os.environ.get("PUMP_LOOKBACK","540"))  # 90d on 4H
PUMP_DROP_PCT = float(os.environ.get("PUMP_DROP_PCT","0.50")) # <50% of recent high

# ATR-based SL (replaces fixed SL_BUFFER)
ATR_PERIOD    = int(os.environ.get("ATR_PERIOD",   "14"))
ATR_MULT      = float(os.environ.get("ATR_MULT",   "1.5"))   # SL = candle_high + 1.5×ATR

# Volume filter: bounce vol must be < VOL_RATIO × prior vol (weak dead cat)
VOL_RATIO     = float(os.environ.get("VOL_RATIO",  "0.80"))  # bounce must use <80% prior vol
VOL_LOOKBACK  = int(os.environ.get("VOL_LOOKBACK", "10"))    # bars to measure prior vol

TP1_PCT       = float(os.environ.get("TP1_PCT",    "0.20"))  # 20% below entry (60% of pos)
TP2_PCT       = float(os.environ.get("TP2_PCT",    "0.40"))  # 40% below entry (40% of pos)
MAX_HOLD_BARS = int(os.environ.get("MAX_HOLD_BARS", "90"))   # 15 days on 4H
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.015"))

# 10 post-pump dying tokens — confirmed on Binance UM Futures
PAIRS = [
    "PIXEL/USDT:USDT",   # pumped Feb 2024, ~95% off peak
    "PORTAL/USDT:USDT",  # pumped Feb 2024, ~95% off peak
    "AEVO/USDT:USDT",    # pumped Mar 2024, ~90% off peak
    "ALT/USDT:USDT",     # pumped Jan 2024, ~90% off peak
    "BOME/USDT:USDT",    # pumped Mar 2024, ~90% off peak
    "MEME/USDT:USDT",    # pumped Oct 2023, ~85% off peak
    "PYTH/USDT:USDT",    # pumped Jan 2024, ~80% off peak
    "DYM/USDT:USDT",     # pumped Jan 2024, ~90% off peak
    "STRK/USDT:USDT",    # pumped Feb 2024, ~85% off peak
    "ESPORTS/USDT:USDT", # pumped Apr 2024, ~95% off peak
]

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─── Data ────────────────────────────────────────────────────────────────────

def cache_path(symbol: str, tf: str, days: int) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return DATA_DIR / f"{safe}_{tf}_{days}d.csv"


def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, tf: str, days: int) -> pd.DataFrame:
    cp = cache_path(symbol, tf, days)
    if cp.exists():
        age_h = (time.time() - cp.stat().st_mtime) / 3600
        if age_h < 12:
            df = pd.read_csv(cp)
            df.columns = ["ts", "open", "high", "low", "close", "volume"]
            return df

    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    since    = start_ms
    all_bars = []
    limit    = 500

    while since < now_ms:
        try:
            bars = ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        except Exception as e:
            print(f"  [fetch error] {e}")
            break
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if len(bars) < limit:
            break
        tf_ms = {"4h": 14_400_000, "1h": 3_600_000, "1d": 86_400_000}.get(tf, 14_400_000)
        since = last_ts + tf_ms
        time.sleep(0.1)

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_csv(cp, index=False)
    return df


# ─── Indicators ──────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]    = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"]    = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["recent_high"] = df["high"].rolling(PUMP_LOOKBACK, min_periods=1).max()

    # ATR(14)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    return df


# ─── Signal ──────────────────────────────────────────────────────────────────

def check_signal(df: pd.DataFrame, i: int) -> bool:
    """
    1. Post-pump: close < 50% of 90d high AND close < EMA50
    2. Trend: last CONFIRM_BARS closes below EMA20 (in downtrend)
    3. Rejection: high >= EMA20 AND close < EMA20 AND close < open (bearish rejection candle)
    """
    min_i = max(EMA_SLOW, CONFIRM_BARS) + 5
    if i < min_i:
        return False

    row        = df.iloc[i]
    close      = float(row["close"])
    high       = float(row["high"])
    open_      = float(row["open"])
    ema_fast   = float(row["ema_fast"])
    ema_slow   = float(row["ema_slow"])
    recent_hi  = float(row["recent_high"])

    # 1. Post-pump filter
    post_pump  = close < recent_hi * PUMP_DROP_PCT and close < ema_slow

    # 2. Prior bars all closed below EMA_FAST (downtrend established)
    prev_closes  = df["close"].iloc[i - CONFIRM_BARS : i].values
    prev_emas    = df["ema_fast"].iloc[i - CONFIRM_BARS : i].values
    in_downtrend = bool((prev_closes < prev_emas).all())

    # 3. Volume filter: bounce volume < VOL_RATIO × prior dump volume
    #    Dead cats bounce on low conviction (sellers still dominate)
    bounce_vol = float(df["volume"].iloc[i - CONFIRM_BARS : i + 1].mean())
    prior_start = max(0, i - CONFIRM_BARS - VOL_LOOKBACK)
    prior_vol   = float(df["volume"].iloc[prior_start : i - CONFIRM_BARS].mean())
    weak_bounce = prior_vol <= 0 or bounce_vol < prior_vol * VOL_RATIO

    # 4. Rejection candle at EMA_FAST
    touches_ema  = high >= ema_fast * 0.995  # 0.5% tolerance
    closes_below = close < ema_fast
    bearish      = close < open_

    return post_pump and in_downtrend and weak_bounce and touches_ema and closes_below and bearish


# ─── Simulation ───────────────────────────────────────────────────────────────

def run_sim(df: pd.DataFrame) -> list[dict]:
    trades = []
    n      = len(df)
    i      = max(EMA_SLOW, PUMP_LOOKBACK) + CONFIRM_BARS + 5

    while i < n:
        if not check_signal(df, i):
            i += 1
            continue

        entry        = float(df["close"].iloc[i])
        candle_hi    = float(df["high"].iloc[i])
        ema_at_entry = float(df["ema_fast"].iloc[i])
        atr_val      = float(df["atr"].iloc[i])

        if entry <= 0 or atr_val <= 0:
            i += 1
            continue

        # ATR-based SL: adapts to each token's volatility
        sl = max(candle_hi, ema_at_entry) + ATR_MULT * atr_val
        tp1   = entry * (1 - TP1_PCT)
        tp2   = entry * (1 - TP2_PCT)

        rr1   = (entry - tp1) / (sl - entry)
        if rr1 < 0.8:   # skip if R:R below 0.8:1
            i += 1
            continue

        exit_bar    = min(i + MAX_HOLD_BARS, n - 1)
        exit_price  = float(df["close"].iloc[exit_bar])
        exit_reason = "TIME"
        tp1_hit     = False
        remaining   = 1.0  # fraction of position still open

        for j in range(i + 1, min(i + MAX_HOLD_BARS + 1, n)):
            hi = float(df["high"].iloc[j])
            lo = float(df["low"].iloc[j])

            # SL check (short: price goes UP against us)
            if hi >= sl:
                exit_price  = sl
                exit_reason = "SL"
                exit_bar    = j
                break

            # TP1 check
            if not tp1_hit and lo <= tp1:
                tp1_hit    = True
                remaining  = 0.40  # close 60%, keep 40%

            # TP2 check (runner)
            if tp1_hit and lo <= tp2:
                # Blended exit: 60% already at TP1, 40% at TP2
                exit_price  = tp2
                exit_reason = "TP2"
                exit_bar    = j
                break
        else:
            if tp1_hit:
                exit_price  = float(df["close"].iloc[exit_bar])
                exit_reason = "TP1+TIME"
            else:
                exit_price  = float(df["close"].iloc[exit_bar])
                exit_reason = "TIME"

        # Blended P&L: 60% at TP1, 40% at final exit (or 100% at SL/TIME)
        if tp1_hit and exit_reason not in ("SL", "TIME"):
            pnl_pct = 0.60 * (entry - tp1) / entry + 0.40 * (entry - exit_price) / entry
        elif tp1_hit and exit_reason == "TP1+TIME":
            pnl_pct = 0.60 * (entry - tp1) / entry + 0.40 * (entry - exit_price) / entry
        else:
            pnl_pct = (entry - exit_price) / entry  # short: profit when price goes down

        sl_risk = (sl - entry) / entry  # % risk on SL hit

        trades.append({
            "entry_bar":   i,
            "entry":       entry,
            "sl":          sl,
            "tp1":         tp1,
            "tp2":         tp2,
            "exit_price":  exit_price,
            "exit_reason": exit_reason,
            "pnl_pct":     pnl_pct,
            "sl_risk_pct": sl_risk,
            "hold_bars":   exit_bar - i,
            "win":         pnl_pct > 0,
        })

        i = exit_bar + 1

    return trades


# ─── Metrics ─────────────────────────────────────────────────────────────────

def summarise(trades: list[dict], days: int) -> dict:
    if not trades:
        return {"n": 0}

    df      = pd.DataFrame(trades)
    wins    = df[df["win"]]
    losses  = df[~df["win"]]
    n       = len(df)
    wr      = len(wins) / n
    gw      = wins["pnl_pct"].sum()    if len(wins)   else 0.0
    gl      = abs(losses["pnl_pct"].sum()) if len(losses) else 1e-9
    pf      = gw / gl

    # Fixed-fractional equity (risk per trade proportional to SL distance)
    equity  = [1.0]
    for _, row in df.iterrows():
        pnl_frac = (row["pnl_pct"] / row["sl_risk_pct"]) * RISK_PER_TRADE
        pnl_frac = max(pnl_frac, -RISK_PER_TRADE * 3)
        equity.append(equity[-1] * (1 + pnl_frac))

    eq       = np.array(equity)
    total_r  = (eq[-1] - 1) * 100
    yrs      = days / 365
    cagr     = ((eq[-1]) ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0
    peak_eq  = np.maximum.accumulate(eq)
    max_dd   = ((peak_eq - eq) / (peak_eq + 1e-12)).max() * 100

    returns  = np.array([(r["pnl_pct"] / r["sl_risk_pct"]) * RISK_PER_TRADE for _, r in df.iterrows()])
    tpy      = n / yrs
    sharpe   = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(tpy) if n > 1 else 0.0

    return {
        "n":         n,
        "tpy":       round(tpy, 1),
        "win_rate":  wr * 100,
        "pf":        pf,
        "cagr":      cagr,
        "max_dd":    max_dd,
        "sharpe":    sharpe,
        "tp2_pct":   (df["exit_reason"] == "TP2").mean()         * 100,
        "tp1_pct":   (df["exit_reason"].isin(["TP1+TIME"])).mean()* 100,
        "sl_pct":    (df["exit_reason"] == "SL").mean()          * 100,
        "time_pct":  (df["exit_reason"] == "TIME").mean()        * 100,
        "avg_hold":  df["hold_bars"].mean(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ex = getattr(ccxt, EXCHANGE)({
        "apiKey": API_KEY, "secret": API_SECRET,
        "options": {"defaultType": "future"},
    })

    print("=" * 72)
    print(f"Dead Cat Bounce SHORT backtest | 4H | {BACKTEST_DAYS}d | {EXCHANGE}")
    print(f"Signal : EMA{EMA_FAST} rejection + post-pump (<{PUMP_DROP_PCT:.0%} of 90d high) + vol<{VOL_RATIO:.0%} prior")
    print(f"Exit   : ATR{ATR_PERIOD}x{ATR_MULT} SL | TP1 -{TP1_PCT:.0%} (60%) | TP2 -{TP2_PCT:.0%} (40%) | {MAX_HOLD_BARS}bar max")
    print("=" * 72)

    results = []

    for sym in PAIRS:
        print(f"  {sym:<24} ...", end=" ", flush=True)
        df = fetch_ohlcv(ex, sym, "4h", BACKTEST_DAYS)
        if df.empty:
            print("NO DATA")
            results.append({"symbol": sym.split("/")[0], "n": 0})
            continue

        df = add_indicators(df)
        trades  = run_sim(df)
        stats   = summarise(trades, BACKTEST_DAYS)
        stats["symbol"] = sym.split("/")[0]
        results.append(stats)

        if stats["n"] == 0:
            print(f"{len(df):,} bars -> 0 trades (no signal)")
        else:
            print(
                f"{len(df):,} bars -> {stats['n']} trades | "
                f"WR {stats['win_rate']:.0f}% | PF {stats['pf']:.2f} | "
                f"CAGR {stats['cagr']:+.1f}% | DD {stats['max_dd']:.1f}%"
            )

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"{'Pair':<8} {'N':>4} {'WR%':>5} {'PF':>6} {'CAGR%':>8} {'DD%':>7} {'TP2%':>6} {'SL%':>5} {'AvgBars':>8}")
    print("-" * 78)
    ranked = sorted(results, key=lambda x: x.get("pf", -1), reverse=True)
    for r in ranked:
        if r.get("n", 0) == 0:
            print(f"  {r['symbol']:<8}  no trades / no data")
            continue
        flag = " << PASS" if r["pf"] >= 1.5 and r["win_rate"] >= 50 and r["cagr"] > 0 else ""
        print(
            f"  {r['symbol']:<8} {r['n']:>4} {r['win_rate']:>5.0f} {r['pf']:>6.2f} "
            f"{r['cagr']:>+8.1f} {r['max_dd']:>7.1f} {r['tp2_pct']:>6.0f} "
            f"{r['sl_pct']:>5.0f} {r['avg_hold']:>8.1f}"
            f"{flag}"
        )
    print("=" * 78)
    print(f"PASS: PF>=1.5 | WR>=50% | CAGR>0%")

    # Save all trades for analysis
    all_trades = []
    for sym in PAIRS:
        try:
            cp = cache_path(sym, "4h", BACKTEST_DAYS)
            if not cp.exists():
                continue
            df = pd.read_csv(cp)
            df.columns = ["ts", "open", "high", "low", "close", "volume"]
            df = add_indicators(df)
            trades = run_sim(df)
            for t in trades:
                t["symbol"] = sym.split("/")[0]
            all_trades.extend(trades)
        except Exception:
            pass

    if all_trades:
        out = Path(__file__).parent / "results"
        out.mkdir(exist_ok=True)
        pd.DataFrame(all_trades).to_csv(out / "dcb_trades.csv", index=False)
        print(f"\nAll trades saved: results/dcb_trades.csv")


if __name__ == "__main__":
    main()
