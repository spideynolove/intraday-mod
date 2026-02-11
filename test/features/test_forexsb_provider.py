import unittest
import tempfile
import struct
import gzip
import os
from datetime import date, datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from intraday.providers.forexsb import ForexSBProvider, _parse_lb, _EPOCH, _RECORD_FMT


def make_lb_bytes(n_records: int = 5) -> bytes:
    records = []
    base_minutes = int((_EPOCH + timedelta(days=0)).timestamp() / 60) if False else 5297370
    for i in range(n_records):
        minutes = base_minutes + i * 30
        o = 140000 + i * 10
        h = o + 50
        l = o - 50
        c = o + 20
        vol = 1000 + i * 100
        spread = 10
        records.append(struct.pack(_RECORD_FMT, minutes, o, h, l, c, vol, spread))
    return b''.join(records)


def make_mock_response(data: bytes) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.content = gzip.compress(data)
    return mock


class TestForexSBProvider(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    @patch('intraday.providers.forexsb.requests.get')
    def test_initialization(self, mock_get):
        mock_get.return_value = make_mock_response(make_lb_bytes())
        provider = ForexSBProvider(
            data_dir=self.temp_dir,
            symbol='EURUSD',
            timeframe='M30',
            date_from=date(2010, 1, 1),
            date_to=date(2010, 12, 31),
        )
        self.assertEqual(provider.symbol, 'EURUSD')
        self.assertEqual(provider.timeframe, 'M30')
        mock_get.assert_called_once()

    @patch('intraday.providers.forexsb.requests.get')
    def test_file_naming(self, mock_get):
        mock_get.return_value = make_mock_response(make_lb_bytes())
        provider = ForexSBProvider(
            data_dir=self.temp_dir,
            symbol='GBPUSD',
            timeframe='H1',
            date_from=date(2010, 6, 1),
            date_to=date(2010, 6, 30),
        )
        self.assertTrue(provider.file_path.endswith('GBPUSD_H1_20100601_20100630.feather'))

    @patch('intraday.providers.forexsb.requests.get')
    def test_cached_file_skips_download(self, mock_get):
        mock_get.return_value = make_mock_response(make_lb_bytes())
        ForexSBProvider(
            data_dir=self.temp_dir, symbol='EURUSD', timeframe='M30',
            date_from=date(2010, 1, 1), date_to=date(2010, 12, 31),
        )
        mock_get.reset_mock()
        ForexSBProvider(
            data_dir=self.temp_dir, symbol='EURUSD', timeframe='M30',
            date_from=date(2010, 1, 1), date_to=date(2010, 12, 31),
        )
        mock_get.assert_not_called()

    def test_parse_lb_bytes(self):
        raw = make_lb_bytes(10)
        df = _parse_lb(raw)
        self.assertEqual(len(df), 10)
        self.assertIn('open', df.columns)
        self.assertIn('close', df.columns)
        self.assertEqual(df.iloc[1]['datetime'] - df.iloc[0]['datetime'], timedelta(minutes=30))

    @patch('intraday.providers.forexsb.requests.get')
    def test_url_format(self, mock_get):
        mock_get.return_value = make_mock_response(make_lb_bytes())
        provider = ForexSBProvider(
            data_dir=self.temp_dir, symbol='EURUSD', timeframe='M15',
            date_from=date(2010, 1, 1), date_to=date(2010, 12, 31),
        )
        url = provider._build_url()
        self.assertIn('EURUSD15', url)
        self.assertIn('dukascopy', url)
        self.assertIn('.lb.gz', url)


if __name__ == '__main__':
    unittest.main()
