#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytz
from datetime import datetime, timezone, timedelta
from intraday.providers.calendar_provider import fetch_calendar

LOCAL_CSV = "/home/hung/Public/Trade/FX/data/economic-calendar.csv"
LOCAL_PART_OUT = "data/economic-calendar-local.csv"
LIVE_PART_OUT = "data/economic-calendar-live.csv"
MERGED_OUT = "data/economic-calendar-utc.csv"

LOCAL_END_DATE = datetime(2021, 4, 26, tzinfo=timezone.utc)
LIVE_START_DATE = datetime(2021, 4, 27, tzinfo=timezone.utc)
LIVE_END_DATE = datetime(2026, 1, 31, tzinfo=timezone.utc)

_VOLATILITY_MAP = {
    "Moderate Volatility Expected": 2,
    "High Volatility Expected": 3,
}
_CURRENCY_MAP = {
    "United States": "USD",
    "Euro Zone": "EUR",
    "United Kingdom": "GBP",
    "Japan": "JPY",
    "Canada": "CAD",
    "Australia": "AUD",
    "Switzerland": "CHF",
    "New Zealand": "NZD",
    "China": "CNY",
    "Germany": "EUR",
}
_NY_TZ = pytz.timezone("America/New_York")
_OUTPUT_COLS = ["datetime", "currency", "impact", "event", "actual", "forecast", "previous"]


def build_local_part() -> pd.DataFrame:
    df = pd.read_csv(LOCAL_CSV)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    df["impact"] = df["Volatility"].map(_VOLATILITY_MAP).fillna(1).astype(int)
    df["currency"] = df["Country"].map(_CURRENCY_MAP)
    df = df.dropna(subset=["currency"])
    df["datetime_ny"] = pd.to_datetime(df["Date"] + " " + df["Time_NY"], errors="coerce")
    df = df.dropna(subset=["datetime_ny"])
    df["datetime"] = df["datetime_ny"].dt.tz_localize(_NY_TZ, ambiguous="NaT").dt.tz_convert("UTC")
    df = df.dropna(subset=["datetime"])
    out = df.rename(columns={"Event_Description": "event", "Actual": "actual",
                              "Forecast": "forecast", "Previous": "previous"})
    return out[_OUTPUT_COLS].copy()


def build_live_part() -> pd.DataFrame:
    chunk = timedelta(days=30)
    all_chunks = []
    current = LIVE_START_DATE
    while current < LIVE_END_DATE:
        end = min(current + chunk, LIVE_END_DATE)
        print(f"  Fetching {current.date()} → {end.date()} ...")
        try:
            chunk_df = fetch_calendar(current, end, source="auto")
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
        except Exception as e:
            print(f"  Warning: {e}")
        current = end + timedelta(seconds=1)
    if not all_chunks:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    df = pd.concat(all_chunks, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    if "event" in df.columns:
        df["event"] = df["event"].str.strip()
    for col in _OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    return df[_OUTPUT_COLS].copy()


def main():
    Path("data").mkdir(parents=True, exist_ok=True)

    print("Building local part (2011–2021)...")
    local_df = build_local_part()
    local_df.to_csv(LOCAL_PART_OUT, index=False)
    print(f"  Saved {len(local_df)} events to {LOCAL_PART_OUT}")

    print("Fetching live part (2021-04-27 → 2026-01-31)...")
    live_df = build_live_part()
    live_df.to_csv(LIVE_PART_OUT, index=False)
    print(f"  Saved {len(live_df)} events to {LIVE_PART_OUT}")

    print("Merging and deduplicating...")
    merged = pd.concat([local_df, live_df], ignore_index=True)
    merged["datetime"] = pd.to_datetime(merged["datetime"], utc=True)
    merged = merged.drop_duplicates(subset=["datetime", "currency", "event"])
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged.to_csv(MERGED_OUT, index=False)
    print(f"  Saved {len(merged)} events to {MERGED_OUT}")
    print(f"  Date range: {merged['datetime'].min()} → {merged['datetime'].max()}")


if __name__ == "__main__":
    main()
