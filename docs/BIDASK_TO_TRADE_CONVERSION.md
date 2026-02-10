# Bid-Ask to Trade Conversion Strategy

## I. Problem Definition

**Architectural Mismatch:**
- **Dukascopy provides**: Bid-Ask quotes (L1 market data)
  - Format: `(datetime, ask, bid, ask_vol, bid_vol)`
  - Represents orderbook top-of-book snapshots

- **System expects**: Executed trades with direction
  - Format: `Trade(datetime, operation='B'/'S', amount, price)`
  - Represents actual transactions

**Impact Areas:**
1. Provider layer: Must convert bid-ask → trades
2. Frame.update(): Expects 'B'/'S' operation to estimate spread
3. Features: Many rely on buy/sell imbalance metrics

---

## II. Conversion Strategies

### Strategy A: Tick Rule (Recommended)

**Method**: Infer trade direction from price movement

```python
def convert_bidask_to_trades(ticks):
    trades = []
    prev_mid = None

    for tick in ticks:
        mid_price = (tick.ask + tick.bid) / 2

        if prev_mid is None:
            operation = 'B' if tick.ask_vol > tick.bid_vol else 'S'
            price = tick.ask if operation == 'B' else tick.bid
            volume = tick.ask_vol if operation == 'B' else tick.bid_vol
        else:
            if mid_price > prev_mid:
                operation = 'B'
                price = tick.ask
                volume = tick.ask_vol
            elif mid_price < prev_mid:
                operation = 'S'
                price = tick.bid
                volume = tick.bid_vol
            else:
                operation = 'B' if tick.ask_vol > tick.bid_vol else 'S'
                price = tick.ask if operation == 'B' else tick.bid
                volume = tick.ask_vol if operation == 'B' else tick.bid_vol

        trades.append(Trade(
            datetime=tick.datetime,
            operation=operation,
            amount=volume,
            price=price
        ))
        prev_mid = mid_price

    return trades
```

**Rationale:**
- Price increase → aggressor likely bought (crossed spread, paid ask)
- Price decrease → aggressor likely sold (crossed spread, hit bid)
- No change → use volume imbalance as tiebreaker

**Accuracy**: 60-70% (industry standard for tick rule)

**Pros:**
- Simple, fast
- Preserves price movement dynamics
- No data loss

**Cons:**
- Not true trade direction (inference)
- Can misclassify in choppy markets

---

### Strategy B: Mid-Price with Volume-Weighted

**Method**: Use mid-price for all trades, weight by total volume

```python
def convert_bidask_midprice(ticks):
    trades = []

    for i, tick in enumerate(ticks):
        mid_price = (tick.ask + tick.bid) / 2
        total_volume = tick.ask_vol + tick.bid_vol

        if total_volume > 0:
            operation = 'B'
            trades.append(Trade(
                datetime=tick.datetime,
                operation=operation,
                amount=total_volume,
                price=mid_price
            ))

    return trades
```

**Rationale:**
- Simplest conversion
- Loses buy/sell direction entirely
- Frame.update() will show zero imbalance

**Pros:**
- Fastest, simplest
- Accurate OHLC (using mid-price)
- No inference errors

**Cons:**
- Loses market microstructure information
- Features relying on imbalance become useless
- Spread estimation won't work

---

### Strategy C: Dual Trades (Most Accurate)

**Method**: Generate both bid and ask trades for every tick

```python
def convert_bidask_dual(ticks):
    trades = []

    for tick in ticks:
        if tick.bid_vol > 0:
            trades.append(Trade(
                datetime=tick.datetime,
                operation='S',
                amount=tick.bid_vol,
                price=tick.bid
            ))

        if tick.ask_vol > 0:
            trades.append(Trade(
                datetime=tick.datetime,
                operation='B',
                amount=tick.ask_vol,
                price=tick.ask
            ))

    return sorted(trades, key=lambda t: t.datetime)
```

**Rationale:**
- Preserves full orderbook snapshot
- Both sides of the market represented
- Spread naturally captured (Frame.update() line 154-162 works correctly)

**Pros:**
- Most complete data representation
- Spread calculation accurate (uses consecutive B/S trades)
- Buy/sell imbalances preserved
- Works with existing Frame.update() logic

**Cons:**
- Doubles number of trades (2x processing time)
- Bid/ask volumes not actual trade volumes (just quote sizes)

---

### Strategy D: Hybrid Tick Rule + Dual

**Method**: Use tick rule for direction, but emit both sides

```python
def convert_bidask_hybrid(ticks):
    trades = []
    prev_mid = None

    for tick in ticks:
        mid_price = (tick.ask + tick.bid) / 2

        if prev_mid is None or mid_price == prev_mid:
            primary_op = 'B' if tick.ask_vol >= tick.bid_vol else 'S'
        elif mid_price > prev_mid:
            primary_op = 'B'
        else:
            primary_op = 'S'

        if primary_op == 'B':
            trades.append(Trade(tick.datetime, 'B', tick.ask_vol, tick.ask))
            if tick.bid_vol > 0:
                trades.append(Trade(tick.datetime, 'S', tick.bid_vol * 0.5, tick.bid))
        else:
            trades.append(Trade(tick.datetime, 'S', tick.bid_vol, tick.bid))
            if tick.ask_vol > 0:
                trades.append(Trade(tick.datetime, 'B', tick.ask_vol * 0.5, tick.ask))

        prev_mid = mid_price

    return trades
```

**Rationale:**
- Primary trade reflects inferred direction
- Secondary trade (reduced volume) maintains spread
- Balances accuracy and microstructure preservation

**Pros:**
- Spread estimation works
- Better direction inference than Strategy C
- Imbalances reflect market movement

**Cons:**
- More complex
- Still not "real" trades

---

## III. Impact Analysis

### Frame Fields Affected by Conversion Strategy

| Field | Strategy A | Strategy B | Strategy C | Strategy D |
|-------|------------|------------|------------|------------|
| **OHLC** | ✅ Accurate | ✅ Accurate | ✅ Accurate | ✅ Accurate |
| **volume** | ✅ Preserved | ✅ Preserved | ⚠️ Doubled | ⚠️ 1.5x inflated |
| **avg_trade_spread** | ⚠️ Estimated | ❌ Zero | ✅ Exact | ✅ Exact |
| **buy_ticks/sell_ticks** | ⚠️ Inferred | ❌ All buy | ✅ Both sides | ⚠️ Weighted |
| **imbalance_volume** | ⚠️ Inferred | ❌ Zero | ✅ Preserved | ✅ Weighted |
| **buy_vwap/sell_vwap** | ⚠️ Inferred | ❌ Same | ✅ Accurate | ✅ Accurate |

### Feature Compatibility

**Strategy A (Tick Rule):**
- ✅ Works: OHLC-based features (90% of features)
- ⚠️ Approximate: Imbalance-based features (PriceDynamics, AverageTrade)
- ⚠️ Degraded: Spread-based features (estimated spread less accurate)

**Strategy B (Mid-Price):**
- ✅ Works: OHLC-based features
- ❌ Broken: All imbalance features
- ❌ Broken: Spread features

**Strategy C (Dual Trades):**
- ✅ Works: All features
- ⚠️ Volume metrics need normalization (divide by 2)

**Strategy D (Hybrid):**
- ✅ Works: All features
- ⚠️ Volume metrics need normalization (divide by 1.5)

---

## IV. Recommended Implementation

### Primary: Strategy C (Dual Trades)

**Rationale:**
1. Preserves all market microstructure information
2. Spread calculation works out-of-box
3. Imbalance metrics remain meaningful
4. Minimal code changes needed
5. Volume inflation easily handled via normalization

**Implementation Location**: `intraday/providers/dukascopy_utils.py`

```python
from datetime import datetime, timedelta
from typing import List, Tuple
from intraday.provider import Trade

def normalize_dukascopy_ticks(
    symbol: str,
    day: date,
    hour: int,
    ticks: List[Tuple[int, int, int, float, float]]
) -> List[Trade]:
    base_datetime = datetime(day.year, day.month, day.day, hour, 0, 0)
    point = 100000
    if symbol.lower() in ['usdrub', 'xagusd', 'xauusd']:
        point = 1000

    trades = []
    for time_ms, ask_int, bid_int, ask_vol, bid_vol in ticks:
        dt = base_datetime + timedelta(milliseconds=time_ms)
        ask_price = ask_int / point
        bid_price = bid_int / point

        ask_volume = round(ask_vol * 1000000)
        bid_volume = round(bid_vol * 1000000)

        if bid_volume > 0:
            trades.append(Trade(
                datetime=dt,
                operation='S',
                amount=bid_volume,
                price=bid_price
            ))

        if ask_volume > 0:
            trades.append(Trade(
                datetime=dt,
                operation='B',
                amount=ask_volume,
                price=ask_price
            ))

    return sorted(trades, key=lambda t: (t.datetime, t.operation))
```

**Frame.update() Compatibility**: Already handles consecutive B/S trades

**Volume Normalization**: Add optional parameter to environment

```python
class MultiAgentEnv:
    def __init__(
        self,
        provider,
        # ... existing params ...
        normalize_dual_trades: bool = False
    ):
        self.normalize_dual_trades = normalize_dual_trades
        # If True, divide all volume metrics by 2 for Dukascopy
```

---

### Fallback: Strategy A (Tick Rule)

**Use case**: If volume inflation is unacceptable

**Implementation**:

```python
def normalize_dukascopy_ticks_tickrule(
    symbol: str,
    day: date,
    hour: int,
    ticks: List[Tuple[int, int, int, float, float]]
) -> List[Trade]:
    base_datetime = datetime(day.year, day.month, day.day, hour, 0, 0)
    point = 100000
    if symbol.lower() in ['usdrub', 'xagusd', 'xauusd']:
        point = 1000

    trades = []
    prev_mid = None

    for time_ms, ask_int, bid_int, ask_vol, bid_vol in ticks:
        dt = base_datetime + timedelta(milliseconds=time_ms)
        ask_price = ask_int / point
        bid_price = bid_int / point
        mid_price = (ask_price + bid_price) / 2

        ask_volume = round(ask_vol * 1000000)
        bid_volume = round(bid_vol * 1000000)

        if prev_mid is None:
            operation = 'B' if ask_volume >= bid_volume else 'S'
        elif mid_price > prev_mid:
            operation = 'B'
        elif mid_price < prev_mid:
            operation = 'S'
        else:
            operation = 'B' if ask_volume >= bid_volume else 'S'

        price = ask_price if operation == 'B' else bid_price
        volume = ask_volume if operation == 'B' else bid_volume

        if volume > 0:
            trades.append(Trade(
                datetime=dt,
                operation=operation,
                amount=volume,
                price=price
            ))

        prev_mid = mid_price

    return trades
```

---

## V. Testing Strategy

### Unit Tests: `test/test_bidask_conversion.py`

```python
import unittest
from datetime import date
from intraday.providers.dukascopy_utils import (
    normalize_dukascopy_ticks,
    normalize_dukascopy_ticks_tickrule
)

class TestBidAskConversion(unittest.TestCase):
    def setUp(self):
        self.sample_ticks = [
            (1000, 1095430, 1095420, 1.5, 2.0),
            (2000, 1095450, 1095440, 2.0, 1.0),
            (3000, 1095430, 1095420, 1.0, 1.5),
        ]

    def test_dual_trades_preserves_spread(self):
        trades = normalize_dukascopy_ticks(
            'EURUSD', date(2024, 1, 15), 10, self.sample_ticks
        )

        self.assertEqual(len(trades), 6)

        buy_trades = [t for t in trades if t.operation == 'B']
        sell_trades = [t for t in trades if t.operation == 'S']

        self.assertEqual(len(buy_trades), 3)
        self.assertEqual(len(sell_trades), 3)

        self.assertAlmostEqual(buy_trades[0].price, 1.09543, places=5)
        self.assertAlmostEqual(sell_trades[0].price, 1.09542, places=5)

        spread = buy_trades[0].price - sell_trades[0].price
        self.assertAlmostEqual(spread, 0.00001, places=5)

    def test_tickrule_infers_direction(self):
        trades = normalize_dukascopy_ticks_tickrule(
            'EURUSD', date(2024, 1, 15), 10, self.sample_ticks
        )

        self.assertEqual(len(trades), 3)

        self.assertEqual(trades[0].operation, 'S')
        self.assertEqual(trades[1].operation, 'B')
        self.assertEqual(trades[2].operation, 'S')

    def test_volume_calculation(self):
        trades = normalize_dukascopy_ticks(
            'EURUSD', date(2024, 1, 15), 10, self.sample_ticks
        )

        total_volume = sum(t.amount for t in trades)
        expected = (1.5 + 2.0 + 2.0 + 1.0 + 1.0 + 1.5) * 1000000

        self.assertAlmostEqual(total_volume, expected, places=0)

    def test_zero_volume_handling(self):
        ticks_with_zeros = [
            (1000, 1095430, 1095420, 1.5, 0.0),
            (2000, 1095450, 1095440, 0.0, 2.0),
        ]

        trades = normalize_dukascopy_ticks(
            'EURUSD', date(2024, 1, 15), 10, ticks_with_zeros
        )

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].operation, 'B')
        self.assertEqual(trades[1].operation, 'S')

    def test_datetime_sorting(self):
        trades = normalize_dukascopy_ticks(
            'EURUSD', date(2024, 1, 15), 10, self.sample_ticks
        )

        for i in range(len(trades) - 1):
            self.assertLessEqual(trades[i].datetime, trades[i+1].datetime)
```

### Integration Test: `test/test_dukascopy_frame_generation.py`

```python
from intraday.processor import IntervalProcessor
from intraday.frame import Frame

def test_frame_from_dukascopy_trades():
    ticks = [...]  # Sample Dukascopy ticks
    trades = normalize_dukascopy_ticks('EURUSD', date(2024, 1, 15), 10, ticks)

    processor = IntervalProcessor(method='time', interval=60)
    processor.reset()

    frames = []
    for trade in trades:
        frame = processor.process([trade])
        if frame:
            frames.append(frame)

    assert len(frames) > 0

    frame = frames[0]
    assert frame.open is not None
    assert frame.high >= frame.low
    assert frame.close is not None
    assert frame.avg_trade_spread > 0  # Spread should be detected
    assert frame.buy_ticks > 0
    assert frame.sell_ticks > 0
```

---

## VI. Documentation Updates

### Update `DUKASCOPY_INTEGRATION_PLAN.md`

**Section II.D (Data normalization)** - Replace with:

```markdown
4. **Bid-Ask to Trade Conversion**:
   - Dukascopy provides L1 quotes (bid/ask), not executed trades
   - **Strategy**: Dual trades (emit both bid and ask as separate trades)
   - Bid ticks → Sell trades at bid price
   - Ask ticks → Buy trades at ask price
   - Frame.update() correctly estimates spread from consecutive B/S trades
   - Volume metrics inflated by 2x (acceptable for relative analysis)
   - Alternative: Tick rule (infer direction from price movement, 60-70% accuracy)
```

**Section VII (Open Questions)** - Answer Question 1:

```markdown
1. **Trade operation mapping**: RESOLVED
   - Option B selected (ask=Buy trades, bid=Sell trades)
   - Implementation: Dual trades strategy
   - Both bid and ask emitted as separate trades
   - Preserves spread and microstructure
   - Volume inflation noted in documentation
```

---

## VII. Migration Checklist

- [ ] Create `normalize_dukascopy_ticks()` in dukascopy_utils.py (Dual Trades)
- [ ] Create `normalize_dukascopy_ticks_tickrule()` (Tick Rule fallback)
- [ ] Update `convert_day_archive()` to call normalization function
- [ ] Create unit tests in test/test_bidask_conversion.py
- [ ] Create integration test with Frame generation
- [ ] Update DUKASCOPY_INTEGRATION_PLAN.md Section II.D
- [ ] Add note to DukascopyArchiveProvider docstring about volume inflation
- [ ] Test with real data: verify spread calculation, OHLC accuracy
- [ ] Performance test: ensure 2x trades doesn't cause issues
- [ ] Compare Strategy A vs C with sample data

---

## VIII. Performance Considerations

### Trade Count Impact

**Binance**: ~1000-5000 trades/minute (high liquidity pairs)
**Dukascopy**: ~500-2000 ticks/hour → ~1000-4000 trades/hour with dual strategy

**Impact**: Negligible - Dukascopy has far fewer ticks than Binance trades

### Memory Impact

**Dual trades**: 2x memory for trade list (temporary during conversion)
**Mitigation**: Process hour-by-hour, clear buffers

### Processing Speed

**Test**: 10,000 ticks → 20,000 trades
- Conversion: ~10ms (numpy vectorization possible)
- Frame.update(): ~50ms (existing bottleneck)
- **Total**: <100ms overhead per hour

---

## IX. Alternative: Quote-Based Frame (Future Enhancement)

**Long-term solution**: Add QuoteFrame alongside Frame

```python
class QuoteFrame(Frame):
    def __init__(self):
        super().__init__()
        self.bid_open = None
        self.bid_high = None
        self.bid_low = None
        self.bid_close = None
        self.ask_open = None
        self.ask_high = None
        self.ask_low = None
        self.ask_close = None
        self.spread_avg = None
        self.spread_min = None
        self.spread_max = None

    def update_from_quote(self, quote):
        # Direct bid-ask handling
        pass
```

**Pros:**
- Native bid-ask support
- No conversion needed
- More accurate microstructure

**Cons:**
- Requires refactoring 36 features
- Breaks compatibility with existing code
- Not worth effort for initial implementation

**Recommendation**: Defer to future version if critical

---

## X. Decision Matrix

| Criteria | Strategy A | Strategy B | Strategy C | Strategy D |
|----------|------------|------------|------------|------------|
| Implementation Complexity | Low | Very Low | Low | Medium |
| Compatibility | ⚠️ Partial | ❌ Broken | ✅ Full | ✅ Full |
| Spread Accuracy | ⚠️ Estimated | ❌ Zero | ✅ Exact | ✅ Exact |
| Direction Accuracy | ⚠️ 60-70% | ❌ N/A | ✅ Both sides | ⚠️ Weighted |
| Volume Inflation | ✅ None | ✅ None | ⚠️ 2x | ⚠️ 1.5x |
| Feature Coverage | 80% | 50% | 100% | 95% |
| Performance | Fast | Fastest | Medium | Medium |
| **Recommended** | Fallback | ❌ No | **✅ Primary** | Alternative |

---

## XI. Final Recommendation

**Primary Implementation**: Strategy C (Dual Trades)
- Emit both bid and ask as separate trades
- Preserves all microstructure information
- Works with existing Frame.update() logic
- Document volume inflation (acceptable for relative analysis)

**Fallback Implementation**: Strategy A (Tick Rule)
- If volume inflation becomes problematic
- Provides reasonable approximation
- Loses some microstructure fidelity

**Implementation Steps**:
1. Implement both strategies in dukascopy_utils.py
2. Use Strategy C by default
3. Add parameter to switch to Strategy A if needed
4. Test both with real Dukascopy data
5. Document trade-offs in provider docstring

**Code Change Impact**: ~100 lines new code, zero changes to existing Frame/Feature logic
