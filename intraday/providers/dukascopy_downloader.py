from __future__ import annotations
import io
import lzma
import struct
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
import zstandard as zstd

_BASE_URL = "http://datafeed.dukascopy.com/datafeed"
_RECORD_FMT = ">3i2f"
_RECORD_SIZE = struct.calcsize(_RECORD_FMT)
_PRICE_SCALE = 100000.0


def decode_bi5(data: bytes, day: date) -> list[dict]:
    if not data:
        return []
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError:
        return []
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    n = len(raw) // _RECORD_SIZE
    rows = []
    for i in range(n):
        ms, ask, bid, ask_vol, bid_vol = struct.unpack_from(_RECORD_FMT, raw, i * _RECORD_SIZE)
        dt = day_start + timedelta(milliseconds=ms)
        rows.append({
            "UTC": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00",
            "AskPrice": ask / _PRICE_SCALE,
            "BidPrice": bid / _PRICE_SCALE,
            "AskVolume": ask_vol / _PRICE_SCALE,
            "BidVolume": bid_vol / _PRICE_SCALE,
        })
    return rows


class DukascopyDownloader:
    def __init__(self, output_dir: str, request_delay: float = 0.1):
        self._output_dir = Path(output_dir)
        self._delay = request_delay

    def _build_url(self, symbol: str, day: date, hour: int) -> str:
        m = day.month - 1
        return f"{_BASE_URL}/{symbol}/{day.year}/{m:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"

    def download_day(self, symbol: str, day: date) -> list[dict]:
        all_rows = []
        for hour in range(24):
            url = self._build_url(symbol, day, hour)
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                all_rows.extend(decode_bi5(r.content, day))
            except Exception:
                pass
            if self._delay:
                time.sleep(self._delay)
        return all_rows

    def _write_zst(self, df: pd.DataFrame, path: Path) -> None:
        cctx = zstd.ZstdCompressor(level=3)
        with open(path, "wb") as fh:
            fh.write(cctx.compress(df.to_csv(index=False).encode()))

    def download_year(self, symbol: str, year: int) -> Path:
        return self.download_date_range(
            symbol,
            date(year, 1, 1),
            date(year, 12, 31),
            f"{symbol}_tick_UTC+0_00_{year}-Parse.csv.zst",
        )

    def download_date_range(self, symbol: str, start: date, end: date, filename: str) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / filename
        if out_path.exists():
            return out_path
        all_rows = []
        current = start
        while current <= end:
            rows = self.download_day(symbol, current)
            all_rows.extend(rows)
            current += timedelta(days=1)
        if not all_rows:
            return out_path
        df = pd.DataFrame(all_rows)
        self._write_zst(df, out_path)
        return out_path

    def fill_gap(self, symbol: str, from_year: int, to_date: date) -> list[Path]:
        paths = []
        for year in range(from_year, to_date.year):
            paths.append(self.download_year(symbol, year))
        if to_date.year >= from_year:
            partial_end = date(to_date.year, to_date.month, to_date.day)
            fname = f"{symbol}_tick_UTC+0_00_{to_date.year}-Parse.csv.zst"
            paths.append(self.download_date_range(
                symbol,
                date(to_date.year, 1, 1),
                partial_end,
                fname,
            ))
        return paths
