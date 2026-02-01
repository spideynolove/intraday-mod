# Dukascopy Integration Plan

## I. API Validation Test (Phase 1)

### Objective
Verify Dukascopy API is accessible and data can be downloaded/decompressed.

### Test Script: `test/test_dukascopy_api.py`

**Test targets:**
1. Download single hour bi5 file from Dukascopy
2. Decompress LZMA format (bi5 → binary tokens)
3. Parse binary tokens to tick data (datetime, ask, bid, volumes)
4. Validate data structure matches expected format

**API Endpoint Pattern:**
```
https://www.dukascopy.com/datafeed/{symbol}/{year}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5
```

**Key differences from Binance:**
- **Granularity**: Hourly files (not monthly archives)
- **Format**: bi5 (LZMA compressed binary), not ZIP/CSV
- **Month indexing**: month-1 (0-indexed)
- **Data structure**: 20-byte binary tokens (time_ms, ask, bid, ask_vol, bid_vol)

**Test cases:**
```python
# Test 1: Recent data (EURUSD, yesterday)
# Test 2: Historical data (EURUSD, 2024-01-15)
# Test 3: Different symbol (GBPUSD)
# Test 4: Edge case - DST transition day
# Test 5: Weekend day (should return empty/404)
```

**Success criteria:**
- HTTP 200 response
- Valid LZMA decompression
- Parsed ticks: datetime in UTC, prices as floats, volumes as integers
- Non-empty tick list (500-5000 ticks per hour typical)

---

## II. Implementation Plan (Phase 2)

### A. Architecture Analysis

**BinanceArchiveProvider pattern:**
```
├── __init__: Setup date range, symbol, data_dir
├── Constructor loop: Check/download monthly archives
├── download_month_archive: Fetch ZIP from Binance Vision
├── convert_month_archive: ZIP → Feather (pandas)
├── reset(): Load month file, set trade_index
├── __next__(): Yield Trade namedtuples
└── close(): Clear dataframe
```

**DukascopyArchiveProvider pattern (adapted):**
```
├── __init__: Setup date range, symbol, data_dir
├── Constructor loop: Build file manifest (daily granularity)
├── download_day_archive: Fetch 24 hourly bi5 files
├── convert_day_archive: bi5 → Feather (decompress, parse, concat 24 hours)
├── reset(): Load day file, set trade_index
├── __next__(): Yield Trade namedtuples
└── close(): Clear dataframe
```

**Key architectural decisions:**

1. **Storage granularity**: Daily feather files (not hourly)
   - Rationale: Match BinanceArchiveProvider's monthly pattern at smaller scale
   - Trade-off: Larger files (~10-50MB/day) vs simpler file management

2. **Download strategy**: Parallel hourly downloads using ThreadPoolExecutor
   - Rationale: 24 small requests faster than sequential
   - From duka codebase: concurrent.futures pattern (line 186-196)

3. **Caching**: Convert bi5 → feather once, reuse
   - Rationale: LZMA decompression expensive (100-500ms/hour)
   - Match Binance pattern: check feather exists before downloading

4. **Data normalization**:
   - Point conversion: 100000 (default), 1000 (metals, RUB pairs)
   - Timezone: UTC (Dukascopy serves GMT)
   - DST handling: From duka utils.py (line 647-648)

---

### B. File Structure

```
intraday/providers/
├── dukascopy.py              # DukascopyArchiveProvider
└── dukascopy_utils.py        # Decompression, parsing helpers

test/
├── test_dukascopy_api.py     # Phase 1 validation
└── test_dukascopy_provider.py # Phase 2 integration test
```

---

### C. Implementation Checklist

#### Core Class: `DukascopyArchiveProvider`

**Constructor (`__init__`):**
- [ ] Accept: data_dir, symbol, date_from, date_to
- [ ] Validate inputs (dir exists, symbol format)
- [ ] Build file manifest: `{date: filename}` dict
- [ ] For each day in range:
  - [ ] Check if `{symbol}-{YYYY-MM-DD}.feather` exists
  - [ ] If not, call `download_day_archive` + `convert_day_archive`
- [ ] Store first/last datetime for session boundaries

**Static method: `download_day_archive`**
```python
@staticmethod
def download_day_archive(
    symbol: str,
    day: date,
    file_path_temp: str
) -> List[bytes]:
    # Returns list of 24 bi5 file contents
    # Use ThreadPoolExecutor for parallel downloads
    # Handle DST: skip hour or duplicate hour
```

Implementation notes:
- Month indexing: `month - 1` (Dukascopy uses 0-11)
- Hour range: 0-23 (24 hours)
- DST handling: Use duka's `is_dst()` logic (line 647)
- Retry logic: 3-5 attempts per hour (network failures common)

**Static method: `convert_day_archive`**
```python
@staticmethod
def convert_day_archive(
    symbol: str,
    day: date,
    bi5_files: List[bytes],
    output_path: str
):
    # Decompress each bi5 → binary tokens
    # Parse tokens → (datetime, ask, bid, ask_vol, bid_vol)
    # Normalize: point conversion, hour offset
    # Concat 24 hours → single DataFrame
    # Convert to Trade format: (datetime, 'B'/'S', amount, price)
    # Save as feather
```

Implementation notes:
- Use duka's `decompress_lzma` (line 454-472)
- Use duka's `tokenize` (line 475-481)
- Use duka's `normalize` (line 507-516)
- Hour offset logic from duka's `add_hour` (line 484-504)
- Trade operation: Alternate 'B'/'S' or use mid-price with random direction

**Method: `reset`**
```python
def reset(
    self,
    episode_start_datetime: Union[None, arrow.Arrow, datetime] = None,
    episode_min_duration: Union[None, Real, timedelta] = None,
    seek: Optional[Literal["first", "next", "last"]] = None,
    rng: Optional[np.random.RandomState] = None
) -> datetime:
    # Load feather file for target day
    # Binary search for trade_index
    # Return actual episode start datetime
```

Match BinanceArchiveProvider logic (line 145-251)

**Method: `__next__`**
```python
def __next__(self) -> Trade:
    # Yield current trade
    # Increment trade_index
    # If end of day, call reset(seek="next")
```

Match BinanceArchiveProvider logic (line 253-265)

**Properties:**
- [ ] `name`: Return `{symbol}@Dukascopy`
- [ ] `session_start_datetime`
- [ ] `session_end_datetime`
- [ ] `episode_start_datetime`

---

### D. Helper Module: `dukascopy_utils.py`

Extract from duka codebase:

```python
def decompress_lzma(data: bytes) -> bytes
def tokenize(buffer: bytes) -> List[Tuple[int, int, int, float, float]]
def normalize(symbol: str, day: date, ticks: List) -> List[Tuple[datetime, float, float, int, int]]
def add_hour(ticks: List) -> List
def is_dst(day: date) -> bool
def find_dst_begin(year: int) -> date
def find_dst_end(year: int) -> date
```

**Modifications needed:**
- Remove prints/logging (use intraday Logger pattern)
- Type hints (Python 3.10+)
- Remove duka-specific logic (week start handling)

---

### E. Testing Strategy

**Unit tests (`test/test_dukascopy_provider.py`):**

```python
class TestDukascopyArchiveProvider(unittest.TestCase):
    def test_download_single_day(self):
        # Download EURUSD 2024-01-15
        # Verify feather file created
        # Verify ~20000-50000 ticks

    def test_reset_first_seek(self):
        # Load day, seek="first"
        # Verify episode_start_datetime matches first tick

    def test_iteration(self):
        # Iterate 1000 trades
        # Verify Trade namedtuple structure
        # Verify datetime ascending order

    def test_date_range(self):
        # date_from = 2024-01-10, date_to = 2024-01-12
        # Verify 3 feather files created
        # Reset with random datetime
        # Verify episode in range

    def test_dst_handling(self):
        # Test DST transition day (March 2024)
        # Verify 23 or 25 hours downloaded

    def test_symbol_formats(self):
        # Test EURUSD, XAUUSD (gold - different point)
        # Verify point conversion correct
```

**Integration test:**
```python
def test_with_environment():
    # Create SingleAgentEnv with DukascopyArchiveProvider
    # Run 100 steps
    # Verify state updates, rewards
```

---

### F. Error Handling

**Download failures:**
- Retry 3-5 times with exponential backoff
- If hour unavailable (weekend, data gap): Skip, log warning
- If entire day fails: Raise exception with clear message

**Decompression failures:**
- Log warning, return empty hour
- Continue with remaining hours

**File system:**
- Check disk space before download (24 * 100KB ≈ 2.4MB raw)
- Handle permission errors on data_dir

---

### G. Performance Considerations

**Download optimization:**
- Parallel downloads: ThreadPoolExecutor(max_workers=12)
- Connection pooling: requests.Session()
- Chunk size: 8192 bytes (match duka)

**Memory:**
- Stream downloads (don't load full day in memory)
- Process hour-by-hour
- Clear intermediate buffers

**Storage:**
- Feather format: ~1/3 size of CSV
- Daily file size: 5-30MB (varies by symbol/volatility)
- Typical month: 150-900MB

---

## III. Migration Path from duka Codebase

**Code reuse opportunities:**

1. **Direct copy (minimal changes):**
   - `decompress_lzma` (line 454-472)
   - `tokenize` (line 475-481)
   - `find_sunday`, `find_dst_begin`, `find_dst_end` (line 620-645)
   - `is_dst` (line 647-648)

2. **Adapt logic:**
   - `fetch_day` (line 429-441) → `download_day_archive`
   - `decompress` (line 519-522) → `convert_day_archive`
   - `normalize` (line 507-516) → adjust for Trade format
   - `add_hour` (line 484-504) → integrate into normalize

3. **Discard:**
   - CSV dumper (line 258-356) - use feather
   - CLI app (line 42-74) - out of scope
   - Candle aggregation (line 228-255) - use Processor

---

## IV. Validation Before Full Implementation

### Phase 1 Deliverables (Test First)

**File:** `test/test_dukascopy_api.py`

```python
import requests
import struct
from lzma import LZMADecompressor, FORMAT_AUTO
from datetime import date, datetime, timedelta

def test_download_hour():
    symbol = "EURUSD"
    day = date(2024, 1, 15)
    hour = 10
    url = f"https://www.dukascopy.com/datafeed/{symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"

    response = requests.get(url)
    assert response.status_code == 200
    assert len(response.content) > 0

    print(f"✓ Downloaded {len(response.content)} bytes")

def test_decompress():
    # (use content from test_download_hour)
    decomp = LZMADecompressor(FORMAT_AUTO, None, None)
    decompressed = decomp.decompress(content)

    assert len(decompressed) > 0
    assert len(decompressed) % 20 == 0  # 20-byte tokens

    print(f"✓ Decompressed to {len(decompressed)} bytes, {len(decompressed)//20} ticks")

def test_parse_tokens():
    token_size = 20
    num_tokens = len(decompressed) // token_size
    ticks = []

    for i in range(num_tokens):
        token = decompressed[i*token_size:(i+1)*token_size]
        time_ms, ask, bid, ask_vol, bid_vol = struct.unpack('!IIIff', token)
        ticks.append((time_ms, ask, bid, ask_vol, bid_vol))

    assert len(ticks) > 0
    assert all(0 <= t[0] < 86400000 for t in ticks)  # time_ms in day range

    print(f"✓ Parsed {len(ticks)} ticks")
    print(f"  First tick: {ticks[0]}")
    print(f"  Last tick: {ticks[-1]}")

def test_normalize():
    base_datetime = datetime(2024, 1, 15, 10, 0, 0)  # Hour 10
    normalized = []

    for time_ms, ask, bid, ask_vol, bid_vol in ticks[:10]:
        dt = base_datetime + timedelta(milliseconds=time_ms)
        ask_price = ask / 100000
        bid_price = bid / 100000
        normalized.append({
            'datetime': dt,
            'ask': ask_price,
            'bid': bid_price,
            'ask_volume': round(ask_vol * 1000000),
            'bid_volume': round(bid_vol * 1000000)
        })

    print(f"✓ Normalized sample:")
    for tick in normalized[:3]:
        print(f"  {tick}")
```

**Run test:**
```bash
source /home/hung/env/.venv/bin/activate
cd /home/hung/Public/test/intraday-mod
python test/test_dukascopy_api.py
```

**Expected output:**
```
✓ Downloaded 15234 bytes
✓ Decompressed to 8960 bytes, 448 ticks
✓ Parsed 448 ticks
  First tick: (12000, 1095430, 1095420, 1.5, 0.0)
  Last tick: (3598500, 1095580, 1095570, 2.3, 1.1)
✓ Normalized sample:
  {'datetime': 2024-01-15 10:00:12, 'ask': 1.09543, 'bid': 1.09542, ...}
  {'datetime': 2024-01-15 10:00:45, 'ask': 1.09548, 'bid': 1.09547, ...}
  {'datetime': 2024-01-15 10:01:03, 'ask': 1.09550, 'bid': 1.09549, ...}
```

---

### Phase 2 Deliverables (Implementation)

**After Phase 1 passes:**

1. Create `intraday/providers/dukascopy_utils.py`
2. Create `intraday/providers/dukascopy.py`
3. Update `intraday/providers/__init__.py`:
   ```python
   from .dukascopy import DukascopyArchiveProvider
   ```
4. Create `test/test_dukascopy_provider.py`
5. Run integration test with environment

---

## V. Timeline Estimate

**Phase 1 (API Validation):**
- Test script: 1-2 hours
- Debug/validation: 1 hour
- **Total: 2-3 hours**

**Phase 2 (Implementation):**
- dukascopy_utils.py: 2-3 hours
- DukascopyArchiveProvider: 4-6 hours
- Unit tests: 2-3 hours
- Integration testing: 1-2 hours
- **Total: 9-14 hours**

**Blocked by:** Phase 1 success

---

## VI. Risk Assessment

**High risk:**
- API rate limiting (unknown policy)
  - Mitigation: Add delays between requests, respect 429 responses
- DST handling complexity
  - Mitigation: Extensive testing around March/November transitions

**Medium risk:**
- Weekend data unavailability
  - Mitigation: Skip weekends in file manifest
- Symbol-specific normalization
  - Mitigation: Symbol whitelist with point mappings

**Low risk:**
- Feather format incompatibility
  - Mitigation: Standard pandas, proven in BinanceArchiveProvider

---

## VII. Open Questions (Need User Input)

1. **Trade operation mapping**: Dukascopy gives ask/bid. How to generate 'B'/'S' operation?
   - Option A: Use mid-price, alternate B/S
   - Option B: Use ask=Buy, bid=Sell
   - Option C: Random with 50/50 split
   - **Recommendation**: Option B (ask=Buy trades, bid=Sell trades)

2. **Hour granularity for caching**: Daily vs hourly feather files?
   - Daily: Fewer files, simpler management, 5-30MB each
   - Hourly: Smaller files, slower episode reset
   - **Recommendation**: Daily (matches Binance monthly pattern)

3. **Symbol format**: Accept 'EURUSD' or 'EUR/USD'?
   - **Recommendation**: Accept both, normalize to 'EURUSD' internally

4. **Data range**: Start from what year?
   - Dukascopy has data from ~2003
   - **Recommendation**: Default last 12 months (match Binance), allow override

---

## VIII. Success Criteria

**Phase 1 complete when:**
- [x] All tests in test_dukascopy_api.py pass
- [x] Can download, decompress, parse at least 3 different days
- [x] Tick data structure validated

**Phase 2 complete when:**
- [ ] DukascopyArchiveProvider passes all unit tests
- [ ] Integration test with SingleAgentEnv runs 100+ steps
- [ ] Performance: Episode reset < 500ms, trade iteration < 1ms
- [ ] Documentation: Docstrings for public methods
- [ ] Code review: Passes style/structure review

**Production ready when:**
- [ ] Tested with 5+ different symbols
- [ ] Tested across DST transitions
- [ ] Tested with date ranges spanning months
- [ ] Error handling validated (network failures, missing data)
- [ ] Memory profiling: No leaks over 1000 episodes

---

## IX. Next Steps

**Immediate action:**
1. Create `test/test_dukascopy_api.py` with validation tests
2. Run Phase 1 tests to verify API access
3. Review this plan, answer open questions
4. Proceed to Phase 2 only after Phase 1 success

**User decision required:**
- Approve plan modifications
- Answer open questions (Section VII)
- Confirm test script approach
