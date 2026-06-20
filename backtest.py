"""
backtest.py — Momentum Bot backtest, early-detection signal.

Signal change from v1:
  OLD: 2-bar price ROC >= 3% (entering AFTER pump ran)
  NEW: 1-bar price ROC 0.3-1.5% + vol spike >= 2x (catching FIRST bar of move)

Data cached locally in data/ to avoid re-fetching on each run.

Usage:
  python backtest.py
  EXCHANGE=bybit python backtest.py
  SYMBOL=BTC/USDT:USDT python backtest.py
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
EXCHANGE             = os.environ.get("EXCHANGE",            "binanceusdm")
API_KEY              = os.environ.get("API_KEY",             "")
API_SECRET           = os.environ.get("API_SECRET",          "")

SYMBOL               = os.environ.get("SYMBOL",              "ESPORTS/USDT:USDT")
VOLUME_SPIKE_MIN     = float(os.environ.get("VOLUME_SPIKE_MIN",    "2.0"))
PRICE_ROC_MIN        = float(os.environ.get("PRICE_ROC_MIN",       "0.4"))
PRICE_ROC_MAX        = float(os.environ.get("PRICE_ROC_MAX",       "2.0"))
VOLUME_LOOKBACK_BARS = int(os.environ.get("VOLUME_LOOKBACK_BARS",  "20"))
SL_PCT               = float(os.environ.get("SL_PCT",               "0.015"))
TRAIL_PCT            = float(os.environ.get("TRAIL_PCT",             "0.010"))
MIN_RR               = float(os.environ.get("MIN_RR",                "2.0"))
MAX_HOLD_MINUTES     = int(os.environ.get("MAX_HOLD_MINUTES",        "10"))
LEVERAGE             = int(os.environ.get("LEVERAGE",                "7"))
BACKTEST_DAYS        = int(os.environ.get("BACKTEST_DAYS",           "60"))

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─── Data fetching / caching ─────────────────────────────────────────────────

def cache_path(symbol: str, days: int) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return DATA_DIR / f"{safe}_{days}d_1m.csv"


def fetch_all(ex: ccxt.Exchange, symbol: str, days: int) -> pd.DataFrame:
    cp = cache_path(symbol, days)
    if cp.exists():
        age_h = (time.time() - cp.stat().st_mtime) / 3600
        if age_h < 6:
            print(f"  (cached {age_h:.1f}h ago)")
            return pd.read_csv(cp)

    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    since    = start_ms
    all_bars = []
    limit    = 1000

    while since < now_ms:
        try:
            bars = ex.fetch_ohlcv(symbol, "1m", since=since, limit=limit)
        except Exception as e:
            print(f"\n  [fetch error] {e}")
            break
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if len(bars) < limit:
            break
        since = last_ts + 60_000
        time.sleep(0.08)

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_csv(cp, index=False)
    return df


# ─── Signal (early-detection) ────────────────────────────────────────────────

def check_signal(df: pd.DataFrame, i: int,
                 vol_min=None, roc_min=None, roc_max=None, lb=None) -> bool:
    """
    Single-bar ROC: fires on the FIRST bar of a momentum move, not after 2 bars.
    Thresholds much lower since we're measuring 1 bar, not 2.
    """
    vol_min = vol_min or VOLUME_SPIKE_MIN
    roc_min = roc_min or PRICE_ROC_MIN
    roc_max = roc_max or PRICE_ROC_MAX
    lb      = lb      or VOLUME_LOOKBACK_BARS

    if i < lb + 1:
        return False

    baseline_vol  = df["volume"].iloc[i - lb : i].mean()
    current_vol   = df["volume"].iloc[i]
    vol_spike     = current_vol / max(baseline_vol, 1e-9)

    prev_close    = float(df["close"].iloc[i - 1])
    current_price = float(df["close"].iloc[i])
    if prev_close == 0:
        return False
    price_roc = (current_price - prev_close) / prev_close * 100

    return vol_spike >= vol_min and roc_min <= price_roc <= roc_max


# ─── Simulation ───────────────────────────────────────────────────────────────

def run_sim(df: pd.DataFrame,
            sl_pct=None, trail_pct=None, min_rr=None, max_hold=None,
            vol_min=None, roc_min=None, roc_max=None) -> list[dict]:
    sl_pct    = sl_pct    or SL_PCT
    trail_pct = trail_pct or TRAIL_PCT
    min_rr    = min_rr    or MIN_RR
    max_hold  = max_hold  or MAX_HOLD_MINUTES

    trades = []
    n = len(df)
    i = VOLUME_LOOKBACK_BARS + 2

    while i < n:
        if not check_signal(df, i, vol_min, roc_min, roc_max):
            i += 1
            continue

        entry = float(df["close"].iloc[i])
        if round(entry, 4) == 0:
            i += 1
            continue

        hard_stop  = entry * (1 - sl_pct)
        tp_price   = entry * (1 + sl_pct * min_rr)
        trail_stop = entry * (1 - trail_pct)
        peak       = entry
        exit_bar   = min(i + max_hold, n - 1)
        exit_price = float(df["close"].iloc[exit_bar])
        exit_reason = "TIME"

        for j in range(i + 1, min(i + max_hold + 1, n)):
            hi = float(df["high"].iloc[j])
            lo = float(df["low"].iloc[j])

            if hi > peak:
                peak = hi
                new_trail = peak * (1 - trail_pct)
                if new_trail > trail_stop:
                    trail_stop = new_trail

            effective_stop = max(hard_stop, trail_stop)
            stop_label     = "TRAIL" if trail_stop > hard_stop else "SL"

            if lo <= effective_stop and hi >= tp_price:
                exit_price, exit_reason, exit_bar = effective_stop, stop_label, j
                break
            elif lo <= effective_stop:
                exit_price, exit_reason, exit_bar = effective_stop, stop_label, j
                break
            elif hi >= tp_price:
                exit_price, exit_reason, exit_bar = tp_price, "TP", j
                break
        else:
            exit_price  = float(df["close"].iloc[exit_bar])
            exit_reason = "TIME"

        pnl_pct = (exit_price - entry) / entry

        trades.append({
            "entry_bar":   i,
            "pnl_pct":     pnl_pct,
            "pnl_lev_pct": pnl_pct * LEVERAGE * 100,
            "exit_reason": exit_reason,
            "hold_bars":   exit_bar - i,
            "win":         pnl_pct > 0,
        })
        i = exit_bar + 1

    return trades


# ─── Metrics ─────────────────────────────────────────────────────────────────

def metrics(trades: list[dict], days: int, sl_pct_ref: float | None = None) -> dict:
    if not trades:
        return {"n": 0}

    sl_ref  = sl_pct_ref or SL_PCT
    df      = pd.DataFrame(trades)
    wins    = df[df["win"]]
    losses  = df[~df["win"]]
    n       = len(df)
    wr      = len(wins) / n
    avg_win = wins["pnl_pct"].mean()   if len(wins)   else 0.0
    avg_los = losses["pnl_pct"].mean() if len(losses) else 0.0
    gw      = wins["pnl_pct"].sum()    if len(wins)   else 0.0
    gl      = abs(losses["pnl_pct"].sum()) if len(losses) else 1e-9
    pf      = gw / gl

    # Fixed-fractional equity curve: risk RISK_PER_TRADE per SL hit.
    # pnl as % of portfolio = (price_pct / sl_pct) * RISK_PER_TRADE
    # This prevents multiplicative leverage overflow over many trades.
    RISK_PER_TRADE = 0.015
    returns = np.array([(p / sl_ref) * RISK_PER_TRADE for p in df["pnl_pct"]])
    returns = np.clip(returns, -RISK_PER_TRADE * 4, None)  # floor: max 4× risk loss

    equity  = np.cumprod(1 + np.insert(returns, 0, 0.0))
    eq      = equity
    total_r = (eq[-1] - 1) * 100
    yrs     = days / 365
    cagr    = ((eq[-1]) ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0

    # Drawdown
    peak_eq = np.maximum.accumulate(eq)
    dd_arr  = (peak_eq - eq) / (peak_eq + 1e-12)
    max_dd  = dd_arr.max() * 100

    # Sharpe / Sortino (on fixed-frac returns, annualised by trade freq)
    trades_per_yr = n / yrs
    sharpe  = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(trades_per_yr)
    neg     = returns[returns < 0]
    sortino = (returns.mean() / (neg.std() + 1e-9)) * np.sqrt(trades_per_yr) if len(neg) > 1 else 0.0
    exp_r   = wr * avg_win + (1 - wr) * avg_los  # raw price pct expectancy

    return {
        "n":         n,
        "trades_yr": round(n / yrs, 1),
        "win_rate":  wr * 100,
        "pf":        pf,
        "exp_r":     exp_r * 100,
        "total_ret": total_r,
        "cagr":      cagr,
        "max_dd":    max_dd,
        "sharpe":    sharpe,
        "sortino":   sortino,
        "recov":     total_r / max_dd if max_dd else 0,
        "avg_win":   avg_win * 100,
        "avg_los":   avg_los * 100,
        "tp_pct":    (df["exit_reason"] == "TP"               ).mean() * 100,
        "sl_pct":    (df["exit_reason"].isin(["SL","TRAIL"])  ).mean() * 100,
        "time_pct":  (df["exit_reason"] == "TIME"             ).mean() * 100,
        "avg_hold":  df["hold_bars"].mean(),
    }


def print_metrics(m: dict, symbol: str, days: int):
    print(f"\n{'='*58}")
    print(f" {symbol} | {days}d backtest | {LEVERAGE}x leverage")
    print(f" Signal: vol>={VOLUME_SPIKE_MIN}x + ROC {PRICE_ROC_MIN}-{PRICE_ROC_MAX}% (1-bar)")
    print(f" Exit  : SL {SL_PCT:.1%} | Trail {TRAIL_PCT:.1%} | TP {SL_PCT*MIN_RR:.2%} | {MAX_HOLD_MINUTES}min")
    print(f"{'='*58}")
    if m.get("n", 0) == 0:
        print("  No trades generated.")
        return
    rows = [
        ("Total trades",           f"{m['n']}"),
        ("Trades / year",          f"{m['trades_yr']}"),
        ("Win rate %",             f"{m['win_rate']:.1f}%"),
        ("TP / SL / Time %",       f"{m['tp_pct']:.0f}% / {m['sl_pct']:.0f}% / {m['time_pct']:.0f}%"),
        ("Profit factor",          f"{m['pf']:.3f}"),
        ("Expectancy (% / trade)", f"{m['exp_r']:+.3f}%"),
        ("Total return (lev)",     f"{m['total_ret']:+.1f}%"),
        ("CAGR (lev)",             f"{m['cagr']:+.1f}%"),
        ("Max drawdown (lev)",     f"-{m['max_dd']:.1f}%"),
        ("Sharpe",                 f"{m['sharpe']:.2f}"),
        ("Sortino",                f"{m['sortino']:.2f}"),
        ("Recovery factor",        f"{m['recov']:.2f}"),
        ("Avg win / Avg loss",     f"{m['avg_win']:+.3f}% / {m['avg_los']:+.3f}%"),
        ("Avg hold (bars/min)",    f"{m['avg_hold']:.1f}"),
    ]
    for label, val in rows:
        print(f"  {label:<26} {val}")

    verdict = "FAIL"
    if m["pf"] >= 1.5 and m["win_rate"] >= 40 and m["total_ret"] > 0 and m["n"] >= 20:
        verdict = "PASS"
    elif m["pf"] >= 1.2 and m["total_ret"] > 0:
        verdict = "MARGINAL"
    print(f"\n  VERDICT: {verdict}")
    print("=" * 58)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ex = getattr(ccxt, EXCHANGE)({
        "apiKey": API_KEY, "secret": API_SECRET,
        "options": {"defaultType": "future"},
    })

    print(f"Fetching {SYMBOL} ({BACKTEST_DAYS}d) ...", end=" ", flush=True)
    df = fetch_all(ex, SYMBOL, BACKTEST_DAYS)
    if df.empty:
        print("NO DATA"); return
    print(f"{len(df):,} bars")

    trades = run_sim(df)
    m      = metrics(trades, BACKTEST_DAYS)
    print_metrics(m, SYMBOL, BACKTEST_DAYS)


if __name__ == "__main__":
    main()
