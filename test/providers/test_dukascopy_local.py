import pytest
import zstandard as zstd
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


def test_missing_file_returns_empty(tmp_path):
    provider = DukascopyLocalProvider(data_dir=str(tmp_path), symbol="EURUSD", years=[2018])
    start = provider.reset()
    assert start is None


def test_stop_iteration_after_last(tmp_path):
    data = make_zst_csv(["2018-01-02T01:00:00.000+00:00,1.20010,1.20000,1.5,2.0"])
    f = tmp_path / "EURUSD_tick_UTC+0_00_2018-Parse.csv.zst"
    f.write_bytes(data)
    provider = DukascopyLocalProvider(data_dir=str(tmp_path), symbol="EURUSD", years=[2018])
    provider.reset()
    next(provider)
    next(provider)
    with pytest.raises(StopIteration):
        next(provider)
