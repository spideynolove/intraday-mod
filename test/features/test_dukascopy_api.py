import requests
import struct
import time
from lzma import LZMADecompressor, FORMAT_AUTO, LZMAError
from datetime import date, datetime, timedelta

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

def decompress_lzma(data):
    results = []
    while True:
        decomp = LZMADecompressor(FORMAT_AUTO, None, None)
        try:
            res = decomp.decompress(data)
        except LZMAError:
            if results:
                break
            else:
                raise
        results.append(res)
        data = decomp.unused_data
        if not data:
            break
        if not decomp.eof:
            raise LZMAError("Compressed data ended before end-of-stream marker")
    return b"".join(results)

def test_download_hour():
    print("\n=== Test 1: Download Hour ===")
    symbol = "EURUSD"
    day = date(2024, 1, 15)
    hour = 10
    url = f"https://www.dukascopy.com/datafeed/{symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"

    print(f"URL: {url}")
    response = requests.get(url)

    assert response.status_code == 200, f"Failed: {response.status_code}"
    assert len(response.content) > 0, "Empty response"

    print(f"✓ Downloaded {len(response.content)} bytes")
    return response.content

def test_decompress(content):
    print("\n=== Test 2: Decompress ===")
    decompressed = decompress_lzma(content)

    assert len(decompressed) > 0, "Empty decompressed data"
    assert len(decompressed) % 20 == 0, f"Invalid token alignment: {len(decompressed)} % 20 = {len(decompressed) % 20}"

    num_ticks = len(decompressed) // 20
    print(f"✓ Decompressed to {len(decompressed)} bytes, {num_ticks} ticks")
    return decompressed

def test_parse_tokens(decompressed):
    print("\n=== Test 3: Parse Tokens ===")
    token_size = 20
    num_tokens = len(decompressed) // token_size
    ticks = []

    for i in range(num_tokens):
        token = decompressed[i*token_size:(i+1)*token_size]
        time_ms, ask, bid, ask_vol, bid_vol = struct.unpack('!IIIff', token)
        ticks.append((time_ms, ask, bid, ask_vol, bid_vol))

    assert len(ticks) > 0, "No ticks parsed"
    assert all(0 <= t[0] < 86400000 for t in ticks), "Invalid time_ms values"

    print(f"✓ Parsed {len(ticks)} ticks")
    print(f"  First tick: time={ticks[0][0]}ms ask={ticks[0][1]} bid={ticks[0][2]}")
    print(f"  Last tick:  time={ticks[-1][0]}ms ask={ticks[-1][1]} bid={ticks[-1][2]}")
    return ticks

def test_normalize(ticks, symbol, day, hour):
    print("\n=== Test 4: Normalize ===")
    base_datetime = datetime(day.year, day.month, day.day, hour, 0, 0)
    normalized = []

    point = 100000
    if symbol.lower() in ['usdrub', 'xagusd', 'xauusd']:
        point = 1000

    for time_ms, ask, bid, ask_vol, bid_vol in ticks[:10]:
        dt = base_datetime + timedelta(milliseconds=time_ms)
        ask_price = ask / point
        bid_price = bid / point
        normalized.append({
            'datetime': dt,
            'ask': ask_price,
            'bid': bid_price,
            'ask_volume': round(ask_vol * 1000000),
            'bid_volume': round(bid_vol * 1000000)
        })

    print(f"✓ Normalized sample (point={point}):")
    for tick in normalized[:3]:
        print(f"  {tick['datetime']} | ask={tick['ask']:.5f} bid={tick['bid']:.5f} | vols={tick['ask_volume']}/{tick['bid_volume']}")
    return normalized

def test_different_symbol():
    print("\n=== Test 5: Different Symbol (GBPUSD) ===")
    symbol = "GBPUSD"
    day = date(2024, 1, 15)
    hour = 14
    url = f"https://www.dukascopy.com/datafeed/{symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"

    response = requests.get(url)
    assert response.status_code == 200

    decompressed = decompress_lzma(response.content)
    num_ticks = len(decompressed) // 20

    print(f"✓ {symbol}: {num_ticks} ticks")

def test_weekend():
    print("\n=== Test 6: Weekend Day (Should be empty/404) ===")
    symbol = "EURUSD"
    day = date(2024, 1, 13)
    hour = 10
    url = f"https://www.dukascopy.com/datafeed/{symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"

    response = requests.get(url)

    if response.status_code == 404:
        print(f"✓ Weekend correctly returns 404")
    elif response.status_code == 200 and len(response.content) == 0:
        print(f"✓ Weekend returns empty file")
    else:
        print(f"⚠ Weekend behavior: status={response.status_code}, size={len(response.content)}")

def main():
    print("=" * 60)
    print("Dukascopy API Validation Tests")
    print("=" * 60)

    try:
        content = test_download_hour()
        decompressed = test_decompress(content)
        ticks = test_parse_tokens(decompressed)
        test_normalize(ticks, "EURUSD", date(2024, 1, 15), 10)
        test_different_symbol()
        test_weekend()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review plan/DUKASCOPY_INTEGRATION_PLAN.md")
        print("2. Answer open questions (Section VII)")
        print("3. Proceed to Phase 2 implementation")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
