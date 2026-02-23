from __future__ import annotations
from datetime import datetime, timezone
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional, Type, NamedTuple
import numpy as np
import polars as pl
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
        self._datetimes: np.ndarray | None = None
        self._is_buy: np.ndarray | None = None
        self._amounts: np.ndarray | None = None
        self._prices: np.ndarray | None = None
        self._total: int = 0
        self._index: int = 0
        self._episode_start: Optional[datetime] = None
        self._loaded: bool = False
        self._shm_blocks: dict[str, SharedMemory] = {}

    def _find_file(self, year: int) -> Optional[Path]:
        pattern = f"{self._symbol}_tick_UTC+0_00_{year}-Parse.csv.zst"
        for p in self._data_dir.rglob(pattern):
            return p
        return None

    def _load_year(self, year: int) -> pl.DataFrame | None:
        path = self._find_file(year)
        if path is None:
            return None
        df = pl.read_csv(path, try_parse_dates=True, ignore_errors=True)
        if df["UTC"].dtype != pl.Datetime("us", "UTC"):
            df = df.with_columns(
                pl.col("UTC").str.to_datetime(time_unit="us", time_zone="UTC")
            )
        buys = df.select(
            pl.col("UTC").alias("datetime"),
            pl.lit("B").alias("operation"),
            pl.col("AskVolume").alias("amount"),
            pl.col("AskPrice").alias("price"),
        )
        sells = df.select(
            pl.col("UTC").alias("datetime"),
            pl.lit("S").alias("operation"),
            pl.col("BidVolume").alias("amount"),
            pl.col("BidPrice").alias("price"),
        )
        return pl.concat([buys, sells])

    def _ensure_loaded(self):
        if self._loaded:
            return
        frames = []
        for year in self._years:
            df = self._load_year(year)
            if df is not None:
                frames.append(df)
        if not frames:
            self._datetimes = np.array([], dtype="datetime64[us]")
            self._is_buy = np.array([], dtype=bool)
            self._amounts = np.array([], dtype=np.float64)
            self._prices = np.array([], dtype=np.float64)
            self._total = 0
            self._loaded = True
            return
        combined = pl.concat(frames).sort("datetime", maintain_order=True)
        self._datetimes = combined["datetime"].to_numpy()
        self._is_buy = combined["operation"].to_numpy() == "B"
        self._amounts = combined["amount"].to_numpy().astype(np.float64)
        self._prices = combined["price"].to_numpy().astype(np.float64)
        self._total = len(self._datetimes)
        del combined, frames
        self._loaded = True

    def share_memory(self) -> dict:
        self._ensure_loaded()
        arrays = {
            "datetimes": self._datetimes,
            "is_buy": self._is_buy,
            "amounts": self._amounts,
            "prices": self._prices,
        }
        refs = {}
        for key, arr in arrays.items():
            shm = SharedMemory(create=True, size=max(1, arr.nbytes))
            if arr.size > 0:
                np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
            self._shm_blocks[key] = shm
            refs[key] = {"name": shm.name, "dtype": str(arr.dtype), "shape": arr.shape}
        return refs

    def unlink_shared_memory(self):
        for shm in self._shm_blocks.values():
            shm.close()
            shm.unlink()
        self._shm_blocks = {}

    @classmethod
    def from_shared_memory(
        cls,
        shm_refs: dict,
        symbol: str,
        years: list[int],
    ) -> DukascopyLocalProvider:
        obj = cls.__new__(cls)
        obj._data_dir = Path("/dev/null")
        obj._symbol = symbol.upper()
        obj._years = sorted(years)
        obj._index = 0
        obj._episode_start = None
        obj._loaded = True
        obj._shm_blocks = {}
        for key, ref in shm_refs.items():
            shm = SharedMemory(name=ref["name"])
            arr = np.ndarray(ref["shape"], dtype=np.dtype(ref["dtype"]), buffer=shm.buf)
            arr.flags.writeable = False
            obj._shm_blocks[key] = shm
            setattr(obj, f"_{key}", arr)
        obj._total = len(obj._datetimes)
        return obj

    @staticmethod
    def _to_python_datetime(dt64) -> datetime:
        return dt64.astype("datetime64[us]").item().replace(tzinfo=timezone.utc)

    def reset(self, episode_min_duration=None, rng=None, **kwargs) -> Optional[datetime]:
        self._ensure_loaded()
        if rng is not None and self._total > 0:
            max_start = max(0, self._total // 2)
            self._index = int(rng.integers(0, max_start))
        else:
            self._index = 0
        self._episode_start = self._to_python_datetime(self._datetimes[self._index]) if self._total > 0 else None
        return self._episode_start

    def close(self):
        for shm in self._shm_blocks.values():
            shm.close()
        self._shm_blocks = {}
        self._datetimes = None
        self._is_buy = None
        self._amounts = None
        self._prices = None
        self._total = 0
        self._index = 0
        self._loaded = False

    def __next__(self) -> Trade:
        if self._index >= self._total:
            raise StopIteration
        i = self._index
        self._index += 1
        return Trade(
            datetime=self._to_python_datetime(self._datetimes[i]),
            operation="B" if self._is_buy[i] else "S",
            amount=float(self._amounts[i]),
            price=float(self._prices[i]),
        )

    @property
    def kind(self) -> Type[NamedTuple]:
        return Trade

    @property
    def name(self) -> str:
        return f"DukascopyLocal/{self._symbol}"

    @property
    def session_start_datetime(self) -> Optional[datetime]:
        return self._to_python_datetime(self._datetimes[0]) if self._total > 0 else None

    @property
    def session_end_datetime(self) -> Optional[datetime]:
        return self._to_python_datetime(self._datetimes[-1]) if self._total > 0 else None

    @property
    def episode_start_datetime(self) -> Optional[datetime]:
        return self._episode_start
