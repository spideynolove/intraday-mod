import pytest
import struct
import lzma
from datetime import date
from unittest.mock import patch, MagicMock
from intraday.providers.dukascopy_downloader import DukascopyDownloader, decode_bi5


def make_bi5(records: list[tuple]) -> bytes:
    raw = b"".join(struct.pack(">3i2f", *r) for r in records)
    return lzma.compress(raw)


def test_decode_bi5_basic():
    records = [(3600000, 120010, 120000, 15000, 20000)]
    data = make_bi5(records)
    rows = decode_bi5(data, date(2024, 1, 2))
    assert len(rows) == 1
    assert abs(rows[0]["AskPrice"] - 1.20010) < 1e-5
    assert abs(rows[0]["BidPrice"] - 1.20000) < 1e-5


def test_decode_bi5_empty():
    rows = decode_bi5(b"", date(2024, 1, 2))
    assert rows == []


def test_decode_bi5_corrupt():
    rows = decode_bi5(b"not-lzma-data", date(2024, 1, 2))
    assert rows == []


def test_download_day_calls_24_urls():
    downloader = DukascopyDownloader(output_dir="/tmp/test-dl")
    bi5_data = make_bi5([(0, 120010, 120000, 10000, 10000)])
    mock_response = MagicMock()
    mock_response.content = bi5_data
    mock_response.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_response) as mock_get:
        downloader.download_day("EURUSD", date(2024, 1, 2))
    urls = [call.args[0] for call in mock_get.call_args_list]
    assert len(urls) == 24
    assert any("EURUSD/2024/00/02/00h_ticks.bi5" in u for u in urls)


def test_build_url_month_zero_indexed():
    downloader = DukascopyDownloader(output_dir="/tmp")
    url = downloader._build_url("GBPUSD", date(2024, 3, 15), 9)
    assert "/2024/02/15/09h_ticks.bi5" in url


def test_download_day_skips_failed_hours():
    downloader = DukascopyDownloader(output_dir="/tmp/test-dl", request_delay=0)
    bi5_data = make_bi5([(0, 120010, 120000, 10000, 10000)])
    call_count = 0

    def side_effect(url, timeout):
        nonlocal call_count
        call_count += 1
        if call_count % 3 == 0:
            raise ConnectionError("simulated failure")
        mock = MagicMock()
        mock.content = bi5_data
        mock.raise_for_status = MagicMock()
        return mock

    with patch("requests.get", side_effect=side_effect):
        rows = downloader.download_day("EURUSD", date(2024, 1, 2))
    assert len(rows) > 0
