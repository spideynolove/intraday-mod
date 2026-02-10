from typing import Optional, Union, Literal
import os
import io
import gzip
import struct
import arrow
import requests
import pandas as pd
from datetime import datetime, date, timezone, timedelta
from intraday.provider import Provider, Trade


_EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_RECORD_FMT = '<7I'
_RECORD_SIZE = struct.calcsize(_RECORD_FMT)
_TF_MINUTES = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080}
_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0',
    'Referer': 'https://data.forexsb.com/',
}


def _parse_lb(data: bytes) -> pd.DataFrame:
    n = len(data) // _RECORD_SIZE
    rows = []
    for i in range(n):
        minutes, o, h, l, c, volume, spread = struct.unpack_from(_RECORD_FMT, data, i * _RECORD_SIZE)
        rows.append((_EPOCH + timedelta(minutes=minutes), o, h, l, c, volume, spread))
    return pd.DataFrame(rows, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'spread'])


class ForexSBProvider(Provider):
    def __init__(
        self,
        data_dir: str,
        symbol: str,
        timeframe: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1'] = 'M30',
        date_from: Optional[Union[date, datetime, arrow.Arrow]] = None,
        date_to: Optional[Union[date, datetime, arrow.Arrow]] = None,
        source: Literal['dukascopy', 'fxcm', 'truefx'] = 'dukascopy',
        price_scale: float = 100000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(data_dir, str) and data_dir > '' and os.path.isdir(data_dir)
        self.data_dir = data_dir
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.source = source
        self.price_scale = price_scale

        if date_to is None:
            date_to = arrow.now()
        elif isinstance(date_to, date):
            date_to = datetime.combine(date_to, datetime.max.time(), tzinfo=timezone.utc)
        if isinstance(date_to, datetime):
            date_to = arrow.get(date_to.astimezone(timezone.utc))
        self.date_to = date_to

        if date_from is None:
            date_from = date_to.shift(months=-12)
        elif isinstance(date_from, date):
            date_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
        if isinstance(date_from, datetime):
            date_from = arrow.get(date_from.astimezone(timezone.utc))
        self.date_from = date_from

        self.file_path = os.path.join(
            data_dir,
            f"{self.symbol}_{timeframe}_{date_from.format('YYYYMMDD')}_{date_to.format('YYYYMMDD')}.feather"
        )

        if not os.path.exists(self.file_path):
            self._download_and_save()

        self._df: Optional[pd.DataFrame] = None
        self._trade_index: Optional[int] = None
        self._episode_start_datetime: Optional[Union[datetime, arrow.Arrow]] = None

    def _build_url(self) -> str:
        minutes = _TF_MINUTES[self.timeframe]
        return f"https://data.forexsb.com/datafeed/data/{self.source}/{self.symbol}{minutes}.lb.gz"

    def _download_and_save(self):
        url = self._build_url()
        response = requests.get(url, headers=_HEADERS, timeout=30)
        response.raise_for_status()

        raw = gzip.decompress(response.content)
        df = _parse_lb(raw)
        df['open'] /= self.price_scale
        df['high'] /= self.price_scale
        df['low'] /= self.price_scale
        df['close'] /= self.price_scale
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

        if self.date_from:
            df = df[df['datetime'] >= pd.Timestamp(self.date_from.datetime)]
        if self.date_to:
            df = df[df['datetime'] <= pd.Timestamp(self.date_to.datetime)]

        df.reset_index(drop=True).to_feather(self.file_path)

    def reset(
        self,
        episode_start_datetime: Optional[Union[datetime, arrow.Arrow]] = None,
        episode_max_duration: Optional[float] = None,
    ):
        super().reset(episode_start_datetime, episode_max_duration)

        if self._df is None:
            self._df = pd.read_feather(self.file_path)

        if episode_start_datetime is None:
            self._trade_index = 0
        else:
            if isinstance(episode_start_datetime, datetime):
                episode_start_datetime = arrow.get(episode_start_datetime.astimezone(timezone.utc))
            self._episode_start_datetime = episode_start_datetime
            start = pd.Timestamp(episode_start_datetime.datetime)
            idx = self._df[self._df['datetime'] >= start].index
            self._trade_index = idx[0] if len(idx) else len(self._df)

    def get_next_trade(self) -> Optional[Trade]:
        if self._trade_index >= len(self._df):
            return None

        row = self._df.iloc[self._trade_index]
        dt = arrow.get(row['datetime'])

        if self._episode_start_datetime and self.episode_max_duration:
            elapsed = (dt - self._episode_start_datetime).total_seconds()
            if elapsed > self.episode_max_duration:
                return None

        self._trade_index += 1

        return Trade(
            datetime=dt,
            operation='B',
            amount=float(row['volume']),
            price=float(row['close'])
        )

    def close(self):
        self._df = None
        self._trade_index = None
