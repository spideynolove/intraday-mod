from __future__ import annotations
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Type, NamedTuple
import pandas as pd
import zstandard as zstd
from ..provider import Provider, Trade


class DukascopyLocalProvider(Provider):
    def __init__(
        self,
        data_dir: str,
        symbol: str,
        years: list[int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._data_dir = Path(data_dir)
        self._symbol = symbol.upper()
        self._years = sorted(years)
        self._trades: list[Trade] = []
        self._index: int = 0
        self._episode_start: Optional[datetime] = None

    def _find_file(self, year: int) -> Optional[Path]:
        pattern = f"{self._symbol}_tick_UTC+0_00_{year}-Parse.csv.zst"
        for p in self._data_dir.rglob(pattern):
            return p
        return None

    def _load_year(self, year: int) -> list[Trade]:
        path = self._find_file(year)
        if path is None:
            return []
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as fh:
            raw = dctx.decompress(fh.read())
        df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip")
        df["UTC"] = pd.to_datetime(df["UTC"], utc=True)
        trades = []
        for row in df.itertuples(index=False):
            dt = row.UTC.to_pydatetime()
            trades.append(Trade(datetime=dt, operation="B", amount=row.AskVolume, price=row.AskPrice))
            trades.append(Trade(datetime=dt, operation="S", amount=row.BidVolume, price=row.BidPrice))
        return trades

    def reset(self, episode_min_duration=None, rng=None, **kwargs) -> Optional[datetime]:
        self._trades = []
        for year in self._years:
            self._trades.extend(self._load_year(year))
        self._trades.sort(key=lambda t: t.datetime)
        if rng is not None and len(self._trades) > 0:
            max_start = max(0, len(self._trades) // 2)
            self._index = int(rng.integers(0, max_start))
        else:
            self._index = 0
        self._episode_start = self._trades[self._index].datetime if self._trades else None
        return self._episode_start

    def close(self):
        self._trades = []
        self._index = 0

    def __next__(self) -> Trade:
        if self._index >= len(self._trades):
            raise StopIteration
        trade = self._trades[self._index]
        self._index += 1
        return trade

    @property
    def kind(self) -> Type[NamedTuple]:
        return Trade

    @property
    def name(self) -> str:
        return f"DukascopyLocal/{self._symbol}"

    @property
    def session_start_datetime(self) -> Optional[datetime]:
        return self._trades[0].datetime if self._trades else None

    @property
    def session_end_datetime(self) -> Optional[datetime]:
        return self._trades[-1].datetime if self._trades else None

    @property
    def episode_start_datetime(self) -> Optional[datetime]:
        return self._episode_start
