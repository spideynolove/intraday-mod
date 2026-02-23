#!/usr/bin/env python3
import sys
import time
import io
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_pandas(path: Path) -> int:
    import io
    import numpy as np
    import pandas as pd
    import zstandard as zstd
    from intraday.provider import Trade

    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        raw = dctx.decompress(fh.read())
    df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip")
    df["UTC"] = pd.to_datetime(df["UTC"], utc=True)
    times = pd.DatetimeIndex(df["UTC"]).to_pydatetime()
    ask_prices = df["AskPrice"].to_numpy()
    ask_vols = df["AskVolume"].to_numpy()
    bid_prices = df["BidPrice"].to_numpy()
    bid_vols = df["BidVolume"].to_numpy()
    buys = [Trade(datetime=t, operation="B", amount=av, price=ap)
            for t, av, ap in zip(times, ask_vols, ask_prices)]
    sells = [Trade(datetime=t, operation="S", amount=bv, price=bp)
             for t, bv, bp in zip(times, bid_vols, bid_prices)]
    return len(buys) + len(sells)


def load_polars(path: Path) -> int:
    import polars as pl
    from intraday.provider import Trade

    df = pl.read_csv(path, try_parse_dates=True, ignore_errors=True)
    if df["UTC"].dtype != pl.Datetime("us", "UTC"):
        df = df.with_columns(
            pl.col("UTC").str.to_datetime(time_unit="us", time_zone="UTC")
        )
    times = df["UTC"].to_list()
    ask_prices = df["AskPrice"].to_numpy()
    ask_vols = df["AskVolume"].to_numpy()
    bid_prices = df["BidPrice"].to_numpy()
    bid_vols = df["BidVolume"].to_numpy()
    buys = [Trade(datetime=t, operation="B", amount=av, price=ap)
            for t, av, ap in zip(times, ask_vols, ask_prices)]
    sells = [Trade(datetime=t, operation="S", amount=bv, price=bp)
             for t, bv, bp in zip(times, bid_vols, bid_prices)]
    return len(buys) + len(sells)


def bench(label: str, fn, path: Path, runs: int = 3):
    times = []
    count = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        count = fn(path)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / runs
    print(f"{label:10s}  {avg:.2f}s avg over {runs} runs  ({count:,} trades)")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to a .csv.zst file")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Benchmarking: {path.name}\n")
    t_pandas = bench("pandas", load_pandas, path, args.runs)
    t_polars = bench("polars", load_polars, path, args.runs)

    speedup = t_pandas / t_polars
    print(f"\nPolars is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than pandas")


if __name__ == "__main__":
    main()
