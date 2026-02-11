# DRL FX Trading Agent — Brainstorm Summary

## The Problem

Local tick data (`/home/hung/Public/duka-resources`) is not connected to the intraday-mod
training pipeline. No provider exists to read it. No SB3 training loop exists. No
TickMicrostructure feature exists to surface what makes tick data better than OHLCV.

---

## Data

**Location:** `/home/hung/Public/duka-resources`

**Format:** `{PAIR}_tick_UTC+0_00_{YEAR}-Parse.csv.zst`

**Columns:** `UTC, AskPrice, BidPrice, AskVolume, BidVolume`

**Timestamp format:** ISO 8601 UTC — `2004-01-01T00:00:25.526+00:00`

**Coverage:** 36 pairs (majors, crosses, XAUUSD, XAGUSD), years 2004–2019, ~8.5B rows, 61 GB compressed

**Gap:** 2020–2024 missing — needs filling via the Dukascopy download API

**Download API:** `http://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5`
- Still alive (tested, HTTP only — HTTPS times out)
- Binary format: LZMA-compressed, records `>3i2f` (5 fields big-endian)
- Fields: `[ms_from_midnight, ask*100000, bid*100000, ask_volume, bid_volume]`
- Existing scripts at `/home/hung/Public/duka-resources/resources/linux/` use deprecated
  pandas API (`df.append`, `error_bad_lines`) — need modernization

---

## Data → Pipeline Conversion

The intraday `Exchange` expects `Trade(datetime, operation, amount, price)` namedtuples.

Dukascopy data is **quote tick data** (bid/ask per price update), not directional trades.

**Conversion:** emit two `Trade` records per CSV row:
- `Trade(dt, "B", ask_volume, ask_price)` — ask side
- `Trade(dt, "S", bid_volume, bid_price)` — bid side

This feeds the `Exchange`'s spread EMA estimation correctly (it measures spread between
consecutive B↔S transitions).

---

## Key Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| DRL framework | Stable-Baselines3 | Simple Gymnasium integration, fast iteration |
| Bar timeframe | 5-minute bars | Best signal/noise balance, aligns across pairs |
| Feature stack | Staged: TA-Lib + TickMicrostructure first, SMC later | Lower dim count = stable training; add complexity only if it helps |
| Calendar data | Local CSV (2011–2021) + CalendarProvider (2021+) | Local file is clean, covers training range |
| New vs modify | New file `dukascopy_local.py` | Provider interface is a clean contract, no existing code needs touching |

---

## Why TickMicrostructure

The `Frame` already computes these fields from the tick stream but no feature exposes them:

| Field | Meaning |
|---|---|
| `frame.vwap` | Volume-weighted average price |
| `frame.buy_ticks` / `frame.ticks` | Order flow imbalance (buy pressure ratio) |
| `frame.avg_trade_spread` | Mean bid-ask spread this bar |
| `frame.trade_spread_max` | Spread spike = liquidity withdrawal |
| `frame.flips` | B↔S direction changes = choppiness |

Without this feature, 61 GB of tick data produces the same agent signal as 5-min OHLCV candles.

---

## Staged Experiment Plan

1. **Baseline:** TA-Lib (24 indicators) + TickMicrostructure (~6 outputs) — ~36 dims
2. **Experiment 2:** Add SMC features (~45 dims) → ~81 dims total
3. **Experiment 3:** Add CalendarEvents (~8 dims) → ~89 dims total

Compare each stage by Sharpe ratio on held-out 2018–2019 data.

---

## Economic Calendar

**Local file:** `/home/hung/Public/Trade/FX/data/economic-calendar.csv`
- 44,750 events, 2011–2021
- Columns: `Date, Time_NY, Country, Volatility, Event_Description, Evaluation, Actual, Forecast, Previous`
- Volatility: `Moderate` / `High` (remap to 2/3 for CalendarEvents impact weight)
- Timestamps in New York time — needs UTC conversion
- Gap after 2021-04-26: use CalendarProvider (FXStreet/FXEmpire)
