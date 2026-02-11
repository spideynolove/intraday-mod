# Dukascopy DRL Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect local Dukascopy tick data to a Stable-Baselines3 DRL training loop, with a modernized downloader to fill the 2020–2024 gap.

**Architecture:** New `DukascopyLocalProvider` reads `.csv.zst` files and emits alternating B/S `Trade` namedtuples. A new `TickMicrostructure` feature surfaces Frame's tick-derived stats. A minimal SB3 training script wires `SingleAgentEnv` → PPO. A modernized downloader updates local data.

**Tech Stack:** Python 3.10+, pandas, zstandard, stable-baselines3, gymnasium, intraday-mod

---

## Task 1: DukascopyLocalProvider

**Files:**
- Create: `intraday/providers/dukascopy_local.py`
- Modify: `intraday/providers/__init__.py`
- Test: `tests/providers/test_dukascopy_local.py`

**Step 1: Write the failing test**

```python
import pytest
import pandas as pd
import io
import zstandard as zstd
from datetime import datetime, timezone
from unittest.mock import patch
from intraday.providers.dukascopy_local import DukascopyLocalProvider
from intraday.provider import Trade

def make_zst_csv(rows: list[str]) -> bytes:
    content = "UTC,AskPrice,BidPrice,AskVolume,BidVolume\n" + "\n".join(rows)
    cctx = zstd.ZstdCompressor()
    return cctx.compress(content.encode())

def test_emits_trade_pairs(tmp_path):
    data = make_zst_csv([
        "2018-01-02T01:00:00.000+00:00,1.20010,1.20000,1.5,2.0",
        "2018-01-02T01:00:01.000+00:00,1.20020,1.20010,1.0,1.5",
    ])
    f = tmp_path / "EURUSD_tick_UTC+0_00_2018-Parse.csv.zst"
    f.write_bytes(data)
    provider = DukascopyLocalProvider(data_dir=str(tmp_path), symbol="EURUSD", years=[2018])
    provider.reset()
    trades = [next(provider) for _ in range(4)]
    assert trades[0].operation == "B"
    assert trades[1].operation == "S"
    assert abs(trades[0].price - 1.20010) < 1e-6
    assert abs(trades[1].price - 1.20000) < 1e-6

def test_kind_is_trade():
    provider = DukascopyLocalProvider(data_dir="/tmp", symbol="EURUSD", years=[2018])
    assert provider.kind == Trade

def test_name_includes_symbol():
    provider = DukascopyLocalProvider(data_dir="/tmp", symbol="GBPUSD", years=[2018])
    assert "GBPUSD" in provider.name
```

**Step 2: Run test to verify it fails**

```bash
source /home/hung/env/.venv/bin/activate
pytest tests/providers/test_dukascopy_local.py -v
```
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write minimal implementation**

```python
from __future__ import annotations
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Type, NamedTuple, Union
import numpy as np
import pandas as pd
import zstandard as zstd
from ..provider import Provider, Trade


class DukascopyLocalProvider(Provider):
    def __init__(
        self,
        data_dir: str,
        symbol: str,
        years: list[int],
        episode_min_duration=None,
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
        df = pd.read_csv(
            io.BytesIO(raw),
            parse_dates=["UTC"],
            on_bad_lines="skip",
        )
        df["UTC"] = pd.to_datetime(df["UTC"], utc=True)
        trades = []
        for row in df.itertuples(index=False):
            dt = row.UTC.to_pydatetime()
            trades.append(Trade(datetime=dt, operation="B", amount=row.AskVolume, price=row.AskPrice))
            trades.append(Trade(datetime=dt, operation="S", amount=row.BidVolume, price=row.BidPrice))
        return trades

    def reset(self, episode_min_duration=None, rng=None, **kwargs) -> datetime:
        self._trades = []
        for year in self._years:
            self._trades.extend(self._load_year(year))
        self._trades.sort(key=lambda t: t.datetime)
        if rng is not None and len(self._trades) > 0:
            max_start = max(0, len(self._trades) // 2)
            self._index = int(rng.randint(0, max_start))
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
```

**Step 4: Add to `__init__.py`**

```python
# intraday/providers/__init__.py — add:
from .dukascopy_local import DukascopyLocalProvider
# and add "DukascopyLocalProvider" to __all__
```

**Step 5: Run tests**

```bash
pytest tests/providers/test_dukascopy_local.py -v
```
Expected: all PASS

**Step 6: Commit**

```bash
git add intraday/providers/dukascopy_local.py intraday/providers/__init__.py tests/providers/test_dukascopy_local.py
git commit -m "feat: add DukascopyLocalProvider for local .csv.zst tick files"
```

---

## Task 2: TickMicrostructure Feature

**Files:**
- Create: `intraday/features/tick_microstructure.py`
- Modify: `intraday/features/__init__.py`
- Test: `tests/features/test_tick_microstructure.py`

**Step 1: Write the failing test**

```python
import pytest
import numpy as np
from unittest.mock import MagicMock
from intraday.features.tick_microstructure import TickMicrostructure

def make_frame(close=1.2, vwap=1.199, ticks=100, buy_ticks=60,
               avg_spread=0.00010, spread_max=0.00020):
    f = MagicMock()
    f.close = close
    f.vwap = vwap
    f.ticks = ticks
    f.buy_ticks = buy_ticks
    f.avg_trade_spread = avg_spread
    f.trade_spread_max = spread_max
    f.flips = 20
    return f

def test_output_shape():
    feat = TickMicrostructure()
    frames = [make_frame() for _ in range(10)]
    state = {}
    feat.process(frames, state)
    assert "tick_microstructure" in state
    assert len(state["tick_microstructure"]) == 5

def test_buy_ratio_bounds():
    feat = TickMicrostructure()
    for buy in [0, 50, 100]:
        frames = [make_frame(ticks=100, buy_ticks=buy)]
        state = {}
        feat.process(frames, state)
        ratio = state["tick_microstructure"][0]
        assert 0.0 <= ratio <= 1.0

def test_zero_ticks_safe():
    feat = TickMicrostructure()
    frames = [make_frame(ticks=0, buy_ticks=0)]
    state = {}
    feat.process(frames, state)
    assert not any(np.isnan(v) for v in state["tick_microstructure"])
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/features/test_tick_microstructure.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
from __future__ import annotations
from collections import namedtuple
from typing import Optional
import numpy as np
import gymnasium as gym
from ..feature import Feature

_Output = namedtuple("TickMicrostructure", [
    "buy_tick_ratio",
    "vwap_deviation",
    "spread_mean_norm",
    "spread_expansion",
    "flip_rate",
])


class TickMicrostructure(Feature):
    def __init__(self, write_to: str = "state", **kwargs):
        super().__init__(**kwargs)
        self._write_to = write_to

    @property
    def output(self):
        return _Output

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def process(self, frames, state: dict) -> Optional[dict]:
        frame = frames[-1]
        ticks = frame.ticks or 1
        buy_ratio = frame.buy_ticks / ticks
        vwap = frame.vwap or frame.close
        vwap_dev = (frame.close - vwap) / vwap if vwap != 0 else 0.0
        avg_spread = frame.avg_trade_spread or 1e-8
        spread_norm = avg_spread / frame.close if frame.close != 0 else 0.0
        spread_exp = (frame.trade_spread_max or avg_spread) / avg_spread
        flip_rate = (frame.flips or 0) / ticks
        result = _Output(
            buy_tick_ratio=float(buy_ratio),
            vwap_deviation=float(vwap_dev),
            spread_mean_norm=float(spread_norm),
            spread_expansion=float(spread_exp),
            flip_rate=float(flip_rate),
        )
        if self._write_to == "state":
            state["tick_microstructure"] = result
        return state
```

**Step 4: Add to `__init__.py`**

```python
# intraday/features/__init__.py — add:
from .tick_microstructure import TickMicrostructure
# and add "TickMicrostructure" to __all__
```

**Step 5: Run tests**

```bash
pytest tests/features/test_tick_microstructure.py -v
```
Expected: all PASS

**Step 6: Commit**

```bash
git add intraday/features/tick_microstructure.py intraday/features/__init__.py tests/features/test_tick_microstructure.py
git commit -m "feat: add TickMicrostructure feature (vwap deviation, order flow, spread)"
```

---

## Task 3: Modernized Dukascopy Downloader

**Files:**
- Create: `intraday/providers/dukascopy_downloader.py`
- Modify: `intraday/providers/__init__.py`
- Test: `tests/providers/test_dukascopy_downloader.py`

**Step 1: Write the failing test**

```python
import pytest
import struct
import lzma
from unittest.mock import patch, MagicMock
from intraday.providers.dukascopy_downloader import DukascopyDownloader, decode_bi5

def make_bi5(records: list[tuple]) -> bytes:
    raw = b"".join(struct.pack(">3i2f", *r) for r in records)
    return lzma.compress(raw)

def test_decode_bi5_basic():
    day_start_ms = 0
    records = [(3600000, 120010, 120000, 15000, 20000)]
    data = make_bi5(records)
    from datetime import date, timezone
    rows = decode_bi5(data, date(2024, 1, 2), timezone.utc)
    assert len(rows) == 1
    assert abs(rows[0]["AskPrice"] - 1.20010) < 1e-5
    assert abs(rows[0]["BidPrice"] - 1.20000) < 1e-5

def test_download_day_calls_correct_urls():
    downloader = DukascopyDownloader(output_dir="/tmp/test-dl")
    bi5_data = make_bi5([(0, 120010, 120000, 10000, 10000)])
    mock_response = MagicMock()
    mock_response.content = bi5_data
    mock_response.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_response) as mock_get:
        from datetime import date
        downloader.download_day("EURUSD", date(2024, 1, 2))
    urls = [call.args[0] for call in mock_get.call_args_list]
    assert any("EURUSD/2024/00/02/00h_ticks.bi5" in u for u in urls)
    assert len(urls) == 24
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/providers/test_dukascopy_downloader.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
from __future__ import annotations
import io
import lzma
import struct
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
import zstandard as zstd

_BASE_URL = "http://datafeed.dukascopy.com/datafeed"
_RECORD_FMT = ">3i2f"
_RECORD_SIZE = struct.calcsize(_RECORD_FMT)
_PRICE_SCALE = 100000.0


def decode_bi5(data: bytes, day: date, tz: timezone) -> list[dict]:
    if not data:
        return []
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError:
        return []
    day_start = datetime(day.year, day.month, day.day, tzinfo=tz)
    n = len(raw) // _RECORD_SIZE
    rows = []
    for i in range(n):
        ms, ask, bid, ask_vol, bid_vol = struct.unpack_from(_RECORD_FMT, raw, i * _RECORD_SIZE)
        dt = day_start + timedelta(milliseconds=ms)
        rows.append({
            "UTC": dt.isoformat(),
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
        y = day.year
        m = day.month - 1
        d = day.day
        return f"{_BASE_URL}/{symbol}/{y}/{m:02d}/{d:02d}/{hour:02d}h_ticks.bi5"

    def download_day(self, symbol: str, day: date) -> list[dict]:
        import time
        all_rows = []
        for hour in range(24):
            url = self._build_url(symbol, day, hour)
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                rows = decode_bi5(r.content, day, timezone.utc)
                all_rows.extend(rows)
            except Exception:
                pass
            time.sleep(self._delay)
        return all_rows

    def download_year(self, symbol: str, year: int) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"{symbol}_tick_UTC+0_00_{year}-Parse.csv.zst"
        if out_path.exists():
            return out_path
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        all_rows = []
        current = start
        while current <= end:
            rows = self.download_day(symbol, current)
            all_rows.extend(rows)
            current += timedelta(days=1)
        if not all_rows:
            return out_path
        df = pd.DataFrame(all_rows)
        cctx = zstd.ZstdCompressor(level=3)
        with open(out_path, "wb") as fh:
            fh.write(cctx.compress(df.to_csv(index=False).encode()))
        return out_path
```

**Step 4: Add to `__init__.py`**

```python
from .dukascopy_downloader import DukascopyDownloader
# add "DukascopyDownloader" to __all__
```

**Step 5: Run tests**

```bash
pytest tests/providers/test_dukascopy_downloader.py -v
```
Expected: all PASS

**Step 6: Commit**

```bash
git add intraday/providers/dukascopy_downloader.py intraday/providers/__init__.py tests/providers/test_dukascopy_downloader.py
git commit -m "feat: add DukascopyDownloader (modernized bi5 fetcher, zst output)"
```

---

## Task 4: SB3 Training Script (Baseline: TA-Lib + TickMicrostructure)

**Files:**
- Create: `scripts/train_sb3_baseline.py`
- No test required — script, not library code

**Step 1: Install dependency**

```bash
source /home/hung/env/.venv/bin/activate
uv pip install stable-baselines3
```

**Step 2: Write the training script**

```python
#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent))

from intraday.env import SingleAgentEnv
from intraday.providers.dukascopy_local import DukascopyLocalProvider
from intraday.processor import IntervalProcessor
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward
from intraday.features import (
    WILLR, ATR, ROC, CCI, MFI, ADXR, NATR, STDDEV, SMA,
    TickMicrostructure,
)


def make_env(data_dir: str, symbol: str, years: list[int]):
    provider = DukascopyLocalProvider(data_dir=data_dir, symbol=symbol, years=years)
    processor = IntervalProcessor(method="time", interval=300)
    features = [
        WILLR(period=14),
        ATR(period=14),
        ROC(period=10),
        CCI(period=20),
        MFI(period=14),
        ADXR(period=14),
        NATR(period=14),
        STDDEV(period=20),
        SMA(period=50),
        TickMicrostructure(),
    ]
    action = BuySellCloseAction()
    reward = BalanceReward()
    return SingleAgentEnv(
        provider=provider,
        processor=processor,
        features=features,
        action=action,
        reward=reward,
        episode_min_duration=3600 * 8,
        episode_max_duration=3600 * 24,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/hung/Public/duka-resources")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2012, 2018)))
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--output", default="models/ppo_baseline")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        lambda: make_env(args.data_dir, args.symbol, args.years),
        n_envs=4,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/ppo_baseline")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output)
    env.save(args.output + "_vecnorm.pkl")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 3: Verify it imports cleanly**

```bash
source /home/hung/env/.venv/bin/activate
python scripts/train_sb3_baseline.py --help
```
Expected: prints usage without error

**Step 4: Commit**

```bash
git add scripts/train_sb3_baseline.py
git commit -m "feat: add SB3 PPO baseline training script (TA-Lib + TickMicrostructure)"
```

---

## Task 5: Wire Calendar CSV into CalendarEvents

**Files:**
- Modify: `intraday/features/calendar_events.py` (add UTC conversion for NY timestamps)
- Create: `scripts/build_calendar.py`
- No new tests — existing CalendarEvents tests cover core logic

**Step 1: Write the calendar builder script**

```python
#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from datetime import timezone
import pytz

LOCAL_CSV = "/home/hung/Public/Trade/FX/data/economic-calendar.csv"
OUTPUT = "data/economic-calendar-utc.csv"

VOLATILITY_MAP = {
    "Moderate Volatility Expected": 2,
    "High Volatility Expected": 3,
}

CURRENCY_MAP = {
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

ny_tz = pytz.timezone("America/New_York")

def main():
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LOCAL_CSV)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    df["impact"] = df["Volatility"].map(VOLATILITY_MAP).fillna(1).astype(int)
    df["currency"] = df["Country"].map(CURRENCY_MAP)
    df = df.dropna(subset=["currency"])
    df["datetime_ny"] = pd.to_datetime(df["Date"] + " " + df["Time_NY"], errors="coerce")
    df = df.dropna(subset=["datetime_ny"])
    df["datetime_utc"] = df["datetime_ny"].dt.tz_localize(ny_tz, ambiguous="NaT").dt.tz_convert("UTC")
    df = df.dropna(subset=["datetime_utc"])
    out = df[["datetime_utc", "currency", "impact", "Event_Description", "Actual", "Forecast", "Previous"]].copy()
    out.columns = ["datetime", "currency", "impact", "event", "actual", "forecast", "previous"]
    out.to_csv(OUTPUT, index=False)
    print(f"Written {len(out)} events to {OUTPUT}")

if __name__ == "__main__":
    main()
```

**Step 2: Run it**

```bash
source /home/hung/env/.venv/bin/activate
python scripts/build_calendar.py
```
Expected: `Written ~38000 events to data/economic-calendar-utc.csv`

**Step 3: Verify output**

```bash
head -5 data/economic-calendar-utc.csv
```
Expected: UTC timestamps, USD/EUR/GBP currency codes, impact 2 or 3

**Step 4: Commit**

```bash
git add scripts/build_calendar.py data/economic-calendar-utc.csv
git commit -m "feat: add calendar builder script (NY→UTC conversion, impact remapping)"
```

---

## Running All Tests

```bash
source /home/hung/env/.venv/bin/activate
pytest tests/ -v --tb=short
```
Expected: all existing 156 tests + new provider/feature tests pass

---

## Experiment Sequence (Post-Implementation)

| Experiment | Features | Dims | Eval metric |
|---|---|---|---|
| Baseline | TA-Lib + TickMicrostructure | ~36 | Sharpe on 2018–2019 |
| +Structure | + SMC (5 features) | ~81 | Compare Sharpe |
| +Calendar | + CalendarEvents | ~89 | Compare Sharpe |

Hold-out: always 2018–2019 EURUSD. Train on 2012–2017.
