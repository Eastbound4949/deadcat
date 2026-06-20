"""
optimize.py — Parameter sweep for Momentum Bot early-detection signal.

Sweeps: vol_spike_min x price_roc_min x price_roc_max x sl_pct x max_hold
Uses cached data from backtest.py (runs backtest.py first if cache missing).

Usage: python optimize.py
       SYMBOL=BTC/USDT:USDT python optimize.py
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import backtest as bt

SYMBOL        = os.environ.get("SYMBOL",        "ESPORTS/USDT:USDT")
BACKTEST_DAYS = int(os.environ.get("BACKTEST_DAYS", "60"))
LEVERAGE      = int(os.environ.get("LEVERAGE",       "7"))
MIN_TRADES    = 20   # skip combos with fewer trades (unreliable stats)

# ─── Sweep grid ──────────────────────────────────────────────────────────────
VOL_SPIKE    = [1.5, 2.0, 2.5, 3.0]          # volume spike multiplier
ROC_MIN      = [0.2, 0.4, 0.6, 0.8, 1.0]     # 1-bar ROC lower bound %
ROC_MAX      = [1.0, 1.5, 2.0, 3.0]          # 1-bar ROC upper bound %
SL_PCT       = [0.010, 0.015, 0.020, 0.025]  # hard stop %
MAX_HOLD     = [5, 10, 15]                   # minutes

# Fixed for sweep (set TRAIL_PCT = SL_PCT * 0.7 automatically)
MIN_RR       = 2.0


def sweep(df: pd.DataFrame):
    results = []
    combos  = list(product(VOL_SPIKE, ROC_MIN, ROC_MAX, SL_PCT, MAX_HOLD))
    total   = len(combos)
    print(f"Running {total} combos on {len(df):,} bars ...\n")

    for idx, (vs, rmin, rmax, sl, mh) in enumerate(combos):
        if rmin >= rmax:
            continue
        trail = sl * 0.6  # trail always tighter than SL

        trades = bt.run_sim(df,
                            sl_pct=sl, trail_pct=trail, min_rr=MIN_RR, max_hold=mh,
                            vol_min=vs, roc_min=rmin, roc_max=rmax)

        if len(trades) < MIN_TRADES:
            continue

        m = bt.metrics(trades, BACKTEST_DAYS, sl_pct_ref=sl)

        results.append({
            "vol_spike": vs,
            "roc_min":   rmin,
            "roc_max":   rmax,
            "sl_pct":    sl,
            "max_hold":  mh,
            "n":         m["n"],
            "win_rate":  m["win_rate"],
            "pf":        m["pf"],
            "sharpe":    m["sharpe"],
            "cagr":      m["cagr"],
            "max_dd":    m["max_dd"],
            "tp_pct":    m["tp_pct"],
        })

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{total} done ...", flush=True)

    return pd.DataFrame(results)


def main():
    ex = getattr(ccxt, bt.EXCHANGE)({
        "apiKey": bt.API_KEY, "secret": bt.API_SECRET,
        "options": {"defaultType": "future"},
    })

    print(f"Loading {SYMBOL} data ({BACKTEST_DAYS}d) ...", end=" ", flush=True)
    df = bt.fetch_all(ex, SYMBOL, BACKTEST_DAYS)
    if df.empty:
        print("NO DATA"); return
    print(f"{len(df):,} bars")

    res = sweep(df)

    if res.empty:
        print(f"\nNo combos produced >= {MIN_TRADES} trades. Lower MIN_TRADES or widen grid.")
        return

    res = res.sort_values("sharpe", ascending=False).reset_index(drop=True)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "sweep_results.csv", index=False)

    print(f"\n{'='*88}")
    print(f"TOP 20 by Sharpe  ({len(res)} combos with >={MIN_TRADES} trades)")
    print(f"{'='*88}")
    print(
        f"{'Vol':>5} {'RocMn':>6} {'RocMx':>6} {'SL%':>5} {'Hold':>5} "
        f"{'N':>5} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'CAGR%':>8} {'DD%':>7} {'TP%':>6}"
    )
    print("-" * 88)
    for _, r in res.head(20).iterrows():
        flag = " <<" if r["pf"] >= 1.5 and r["win_rate"] >= 40 and r["cagr"] > 0 else ""
        print(
            f"{r['vol_spike']:>5.1f} {r['roc_min']:>6.2f} {r['roc_max']:>6.2f} "
            f"{r['sl_pct']*100:>5.1f} {r['max_hold']:>5.0f} "
            f"{r['n']:>5.0f} {r['win_rate']:>6.1f} {r['pf']:>6.3f} "
            f"{r['sharpe']:>7.2f} {r['cagr']:>+8.1f} {r['max_dd']:>7.1f} {r['tp_pct']:>6.0f}"
            f"{flag}"
        )
    print("=" * 88)

    # Best combo
    best = res.iloc[0]
    print(f"\nBEST COMBO (Sharpe={best['sharpe']:.2f}):")
    print(f"  vol>={best['vol_spike']}x  ROC {best['roc_min']}-{best['roc_max']}%  "
          f"SL {best['sl_pct']*100:.1f}%  hold {best['max_hold']:.0f}min  "
          f"-> {best['n']:.0f} trades | WR {best['win_rate']:.1f}% | "
          f"PF {best['pf']:.3f} | CAGR {best['cagr']:+.1f}%")
    print(f"\nSaved: {out / 'sweep_results.csv'}")
    print("\nRe-run backtest with best params:")
    print(
        f"  VOLUME_SPIKE_MIN={best['vol_spike']} PRICE_ROC_MIN={best['roc_min']} "
        f"PRICE_ROC_MAX={best['roc_max']} SL_PCT={best['sl_pct']} "
        f"MAX_HOLD_MINUTES={best['max_hold']:.0f} python backtest.py"
    )


if __name__ == "__main__":
    main()
