# TA-Lib Implementation Guide for intraday-mod

## Overview

This guide provides the methodology for implementing TA-Lib indicators missing from intraday-mod. It covers the complete execution flow, design patterns, and category-specific implementations.

---

## I. System Architecture & Callstack

### Complete Execution Flow

```
User Code
    ↓
env.reset()
    ↓
    ├─> provider.reset()
    ├─> processor.reset()
    ├─> action_scheme.reset()
    ├─> reward_scheme.reset()
    └─> for feature in features_pipeline:
            feature.reset()
    ↓
env.step(action)
    ↓
    ├─> action_scheme.process_action(exchange, account, action, time)
    │       └─> Creates orders (MarketOrder, LimitOrder, etc.)
    ↓
    ├─> _get_next_frame()
    │       └─> while True:
    │               ├─> trade = provider.get_next_trade()
    │               ├─> self.trades.append(trade)
    │               ├─> for feature in trades_features:
    │               │       feature.update(self.trades)  ← TradesFeature.update()
    │               ├─> frame = processor.process(self.trades)
    │               └─> if frame: break
    │           ├─> self.frames.append(frame)
    │           └─> return frame, done
    ↓
    ├─> _make_states()
    │       └─> self.state = OrderedDict()
    │           for feature in features_pipeline:
    │               feature.process(self.frames, self.state)  ← Feature.process()
    │           └─> Add account-specific fields (position, roi, etc.)
    ↓
    ├─> Apply idle_penalty if configured
    ├─> Check episode_max_duration
    ├─> Close positions if done
    ├─> Update balances if not instant_balance_update
    ↓
    └─> for account in accounts:
            reward = reward_scheme.get_reward(env, account)
    ↓
return (states, rewards, done, frame)
```

### Key Call Points

1. **Feature.reset()**: Called once at episode start (env.reset())
2. **TradesFeature.update(trades)**: Called on EVERY trade before frame creation
3. **Feature.process(frames, state)**: Called AFTER frame creation to populate state dict

---

## II. Implementation Patterns by Data Dependency

### Pattern A: Frame-Only Indicators (Most Common)

**When to use:** Indicator uses only OHLCV data from current/past frames

**Base class:** `Feature`

**Required methods:**
- `__init__(...)`: Setup parameters, names, spaces
- `reset()`: Clear internal state
- `process(frames, state)`: Calculate indicator value(s)

**Data access:**
```python
frame = frames[-1]           # Current frame
prev_frame = frames[-2]      # Previous frame (check len(frames) > 1)
window = frames[-period:]    # Sliding window
```

**Examples:** RSI, MACD, Bollinger Bands, ADX, Stochastic

---

### Pattern B: Trade-Stream Indicators (Less Common)

**When to use:** Indicator requires trade-by-trade data (tick volume, spread, order flow)

**Base class:** `TradesFeature`

**Required methods:**
- `__init__(...)`: Setup with trades_period parameter
- `reset()`: Clear accumulators
- `update(trades)`: Accumulate data from each trade
- `process(frames, state)`: Output accumulated results to state

**Data access:**
```python
trade = trades[-1]           # Most recent trade
trade.price                  # Trade price
trade.amount                 # Trade size
trade.operation              # 'B' (buy) or 'S' (sell)
trade.datetime               # Trade timestamp
```

**Examples:** VWAP (if using tick data), Volume Profile, Order Flow Imbalance

---

### Pattern C: Cumulative/Stateful Indicators

**When to use:** Indicator maintains running state across frames (EMA, cumulative sum)

**Base class:** `Feature`

**Required methods:**
- `__init__(...)`: Initialize state variables to None/0
- `reset()`: Reset state variables
- `process(frames, state)`: Update state, calculate output

**State management:**
```python
def reset(self):
    self.ema_value = None
    self._n_iter = 0

def process(self, frames, state):
    if self.ema_value is None:
        self.ema_value = frames[-1].close  # Initialize
    else:
        self.ema_value = ... # Update formula
    state[self.names[0]] = self.ema_value
```

**Examples:** EMA, KAMA, Parabolic SAR, ADX

---

### Pattern D: Rolling Window Indicators

**When to use:** Indicator requires N frames of historical data

**Base class:** `Feature`

**Warmup handling:**
```python
def process(self, frames, state):
    if len(frames) < self.period:
        state[self.names[0]] = 0.0  # or None
        return

    window = frames[-self.period:]
    result = self._calculate(window)
    state[self.names[0]] = result
```

**Examples:** SMA, Standard Deviation, Correlation, Linear Regression

---

## III. General Implementation Principles

### 1. Naming Convention

```python
self.names = [f"{indicator_name}_{period}_{source}"]
```

Examples:
- `rsi_14_close`
- `macd_12_26_9_close`
- `bb_upper_20_close`
- `atr_14`

### 2. Gym Space Definition

Always define spaces for state outputs:

```python
if write_to in {"state", "both"}:
    self.spaces = OrderedDict({
        name: gym.spaces.Box(low, high, shape=(1,))
        for name in self.names
    })
```

Common ranges:
- Unbounded: `Box(-math.inf, math.inf, shape=(1,))`
- Non-negative: `Box(0, math.inf, shape=(1,))`
- Percentage: `Box(-1, 1, shape=(1,))`
- Bounded: `Box(0, 100, shape=(1,))` (e.g., RSI)

### 3. Write-To Logic (Standardized Pattern)

```python
if self.write_to_frame:
    setattr(frames[-1], self.names[i], value)
if self.write_to_state:
    state[self.names[i]] = value
```

### 4. Source Field Access

```python
value = getattr(frame, self.source)
```

Common sources: `'close'`, `'high'`, `'low'`, `'open'`, `'volume'`, `'hlc3'`, `'hl2'`

### 5. Error Handling

```python
if not isinstance(value, Real) or math.isnan(value):
    result = 0.0  # or previous value
```

### 6. Warmup Awareness

The environment handles warmup via `warm_up_time` parameter. Features should:
- Return 0.0 or None when insufficient data
- NOT track warmup internally (environment manages this)

---

## IV. Category-Specific Implementation Guides

### A. Momentum Indicators (RSI, MACD, ADX, etc.)

**General Pattern:**
1. Calculate price changes over period
2. Apply smoothing (usually EMA)
3. Normalize or compare directional movements

#### Example: RSI (Relative Strength Index)

```python
class RSI(Feature):
    def __init__(
        self,
        period: int = 14,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state"
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.names = [f'rsi_{period}_{source}']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, 100, shape=(1,))
            })
        self.avg_gain = None
        self.avg_loss = None
        self._n_iter = 0

    def reset(self):
        self.avg_gain = None
        self.avg_loss = None
        self._n_iter = 0

    def process(self, frames, state):
        if len(frames) < 2:
            rsi = 50.0
        else:
            curr_value = getattr(frames[-1], self.source)
            prev_value = getattr(frames[-2], self.source)
            change = curr_value - prev_value
            gain = max(change, 0)
            loss = max(-change, 0)

            if self.avg_gain is None:
                self.avg_gain = gain
                self.avg_loss = loss
            elif self._n_iter < self.period:
                self.avg_gain = (self.avg_gain * self._n_iter + gain) / (self._n_iter + 1)
                self.avg_loss = (self.avg_loss * self._n_iter + loss) / (self._n_iter + 1)
            else:
                alpha = 1.0 / self.period
                self.avg_gain = (1 - alpha) * self.avg_gain + alpha * gain
                self.avg_loss = (1 - alpha) * self.avg_loss + alpha * loss

            if self.avg_loss == 0:
                rsi = 100.0
            else:
                rs = self.avg_gain / self.avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

            self._n_iter += 1

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], rsi)
        if self.write_to_state:
            state[self.names[0]] = rsi

    def __repr__(self):
        return f"RSI(period={self.period}, source={self.source}, write_to={self.write_to})"
```

**Key points:**
- Stateful (maintains avg_gain, avg_loss)
- Uses EMA-like smoothing
- Handles initialization phase (_n_iter < period)
- Output bounded [0, 100]

---

#### Example: MACD (Moving Average Convergence Divergence)

```python
class MACD(Feature):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state"
    ):
        super().__init__(write_to=write_to)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.source = source
        self.names = [
            f'macd_{fast_period}_{slow_period}_{source}',
            f'macd_signal_{signal_period}_{source}',
            f'macd_hist_{fast_period}_{slow_period}_{signal_period}_{source}'
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                for name in self.names
            })

        self.fast_ema = None
        self.slow_ema = None
        self.signal_ema = None
        self.fast_alpha = 2.0 / (fast_period + 1)
        self.slow_alpha = 2.0 / (slow_period + 1)
        self.signal_alpha = 2.0 / (signal_period + 1)
        self._n_iter = 0

    def reset(self):
        self.fast_ema = None
        self.slow_ema = None
        self.signal_ema = None
        self._n_iter = 0

    def process(self, frames, state):
        value = getattr(frames[-1], self.source)

        if self.fast_ema is None:
            self.fast_ema = value
            self.slow_ema = value
            macd_line = 0.0
            signal_line = 0.0
        else:
            self.fast_ema = self.fast_ema * (1 - self.fast_alpha) + value * self.fast_alpha
            self.slow_ema = self.slow_ema * (1 - self.slow_alpha) + value * self.slow_alpha
            macd_line = self.fast_ema - self.slow_ema

            if self.signal_ema is None:
                self.signal_ema = macd_line
            else:
                self.signal_ema = self.signal_ema * (1 - self.signal_alpha) + macd_line * self.signal_alpha
            signal_line = self.signal_ema

        histogram = macd_line - signal_line

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], macd_line)
            setattr(frames[-1], self.names[1], signal_line)
            setattr(frames[-1], self.names[2], histogram)
        if self.write_to_state:
            state[self.names[0]] = macd_line
            state[self.names[1]] = signal_line
            state[self.names[2]] = histogram

        self._n_iter += 1

    def __repr__(self):
        return f"MACD(fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}, source={self.source})"
```

**Key points:**
- Multiple outputs (macd_line, signal_line, histogram)
- Cascaded EMAs
- Three separate state variables

---

### B. Overlap Studies (Bollinger Bands, TEMA, etc.)

**General Pattern:**
1. Calculate moving average (center line)
2. Calculate deviation or envelope
3. Output multiple bands

#### Example: Bollinger Bands

```python
class BollingerBands(Feature):
    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state"
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.num_std = num_std
        self.names = [
            f'bb_upper_{period}_{source}',
            f'bb_middle_{period}_{source}',
            f'bb_lower_{period}_{source}',
            f'bb_width_{period}_{source}'
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                for name in self.names
            })

    def process(self, frames, state):
        if len(frames) < self.period:
            middle = upper = lower = getattr(frames[-1], self.source)
            width = 0.0
        else:
            window = frames[-self.period:]
            values = [getattr(f, self.source) for f in window]
            middle = sum(values) / self.period
            variance = sum((v - middle) ** 2 for v in values) / self.period
            std_dev = math.sqrt(variance)
            upper = middle + self.num_std * std_dev
            lower = middle - self.num_std * std_dev
            width = upper - lower

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], upper)
            setattr(frames[-1], self.names[1], middle)
            setattr(frames[-1], self.names[2], lower)
            setattr(frames[-1], self.names[3], width)
        if self.write_to_state:
            state[self.names[0]] = upper
            state[self.names[1]] = middle
            state[self.names[2]] = lower
            state[self.names[3]] = width

    def __repr__(self):
        return f"BollingerBands(period={self.period}, std={self.num_std}, source={self.source})"
```

**Key points:**
- Rolling window calculation
- Multiple outputs (upper, middle, lower, width)
- Pure function (no state between frames)

---

### C. Volatility Indicators (ATR, NATR)

**General Pattern:**
1. Calculate True Range
2. Apply smoothing (usually EMA)
3. Optionally normalize

#### Example: ATR (Average True Range)

```python
class ATR(Feature):
    def __init__(
        self,
        period: int = 14,
        write_to: Literal["state", "frame", "both"] = "state"
    ):
        super().__init__(write_to=write_to, period=period)
        self.names = [f'atr_{period}']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, math.inf, shape=(1,))
            })
        self.atr_value = None
        self._n_iter = 0
        self.alpha = 2.0 / (period + 1)

    def reset(self):
        self.atr_value = None
        self._n_iter = 0

    def process(self, frames, state):
        frame = frames[-1]

        if len(frames) < 2:
            true_range = frame.high - frame.low
        else:
            prev_close = frames[-2].close
            true_range = max(
                frame.high - frame.low,
                abs(frame.high - prev_close),
                abs(frame.low - prev_close)
            )

        if self.atr_value is None:
            self.atr_value = true_range
        elif self._n_iter < self.period:
            self.atr_value = (self.atr_value * self._n_iter + true_range) / (self._n_iter + 1)
        else:
            self.atr_value = self.atr_value * (1 - self.alpha) + true_range * self.alpha

        if self.write_to_frame:
            setattr(frame, self.names[0], self.atr_value)
        if self.write_to_state:
            state[self.names[0]] = self.atr_value

        self._n_iter += 1

    def __repr__(self):
        return f"ATR(period={self.period}, write_to={self.write_to})"
```

**Key points:**
- Uses True Range calculation
- EMA smoothing
- Always non-negative

---

### D. Statistical Functions (CORREL, BETA, STDDEV, etc.)

**General Pattern:**
1. Collect rolling window of data
2. Apply statistical formula
3. Return scalar result

#### Example: Correlation

```python
class Correlation(Feature):
    def __init__(
        self,
        period: int = 30,
        source1: str = 'close',
        source2: str = 'volume',
        write_to: Literal["state", "frame", "both"] = "state"
    ):
        super().__init__(write_to=write_to, period=period)
        self.source1 = source1
        self.source2 = source2
        self.names = [f'correl_{period}_{source1}_{source2}']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-1, 1, shape=(1,))
            })

    def process(self, frames, state):
        if len(frames) < self.period:
            correl = 0.0
        else:
            window = frames[-self.period:]
            x_values = [getattr(f, self.source1) for f in window]
            y_values = [getattr(f, self.source2) for f in window]

            x_mean = sum(x_values) / self.period
            y_mean = sum(y_values) / self.period

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            x_variance = sum((x - x_mean) ** 2 for x in x_values)
            y_variance = sum((y - y_mean) ** 2 for y in y_values)

            denominator = math.sqrt(x_variance * y_variance)
            correl = numerator / denominator if denominator != 0 else 0.0

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], correl)
        if self.write_to_state:
            state[self.names[0]] = correl

    def __repr__(self):
        return f"Correlation(period={self.period}, source1={self.source1}, source2={self.source2})"
```

**Key points:**
- Rolling window
- Pure calculation (no state)
- Output bounded [-1, 1]

---

### E. Pattern Recognition (Candlestick Patterns)

**General Pattern:**
1. Access last 1-4 frames depending on pattern
2. Calculate body/shadow ratios
3. Check pattern conditions
4. Return -100 (bearish), 0 (none), +100 (bullish)

#### Example: Doji

```python
class Doji(Feature):
    def __init__(
        self,
        body_threshold: float = 0.1,
        write_to: Literal["state", "frame", "both"] = "state"
    ):
        super().__init__(write_to=write_to)
        self.body_threshold = body_threshold
        self.names = ['cdl_doji']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-100, 100, shape=(1,))
            })

    def process(self, frames, state):
        frame = frames[-1]
        body = abs(frame.close - frame.open)
        range_hl = frame.high - frame.low

        if range_hl == 0:
            result = 0
        else:
            body_ratio = body / range_hl
            result = 100 if body_ratio <= self.body_threshold else 0

        if self.write_to_frame:
            setattr(frame, self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result

    def __repr__(self):
        return f"Doji(threshold={self.body_threshold}, write_to={self.write_to})"
```

#### Example: Engulfing Pattern

```python
class Engulfing(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = ['cdl_engulfing']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-100, 100, shape=(1,))
            })

    def process(self, frames, state):
        if len(frames) < 2:
            result = 0
        else:
            curr = frames[-1]
            prev = frames[-2]

            curr_bullish = curr.close > curr.open
            prev_bullish = prev.close > prev.open

            curr_body_top = max(curr.open, curr.close)
            curr_body_bottom = min(curr.open, curr.close)
            prev_body_top = max(prev.open, prev.close)
            prev_body_bottom = min(prev.open, prev.close)

            bullish_engulfing = (
                curr_bullish and not prev_bullish
                and curr_body_bottom < prev_body_bottom
                and curr_body_top > prev_body_top
            )

            bearish_engulfing = (
                not curr_bullish and prev_bullish
                and curr_body_top > prev_body_top
                and curr_body_bottom < prev_body_bottom
            )

            if bullish_engulfing:
                result = 100
            elif bearish_engulfing:
                result = -100
            else:
                result = 0

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result

    def __repr__(self):
        return f"Engulfing(write_to={self.write_to})"
```

**Key points:**
- Uses 2+ frames
- Returns categorical values (-100, 0, +100)
- Stateless

---

### F. Cycle Indicators (Hilbert Transform)

**Note:** Hilbert Transform indicators are mathematically complex and require specialized DSP knowledge. Implementation requires:

1. Discrete Hilbert Transform calculation
2. Phase/amplitude extraction
3. Cycle period estimation

**Recommendation:** Use scipy.signal.hilbert for initial implementation, then optimize.

#### Skeleton Example: HT_DCPERIOD

```python
import numpy as np
from scipy import signal

class HT_DCPERIOD(Feature):
    def __init__(
        self,
        source: str = 'close',
        min_period: int = 6,
        max_period: int = 50,
        write_to: Literal["state", "frame", "both"] = "state"
    ):
        super().__init__(write_to=write_to)
        self.source = source
        self.min_period = min_period
        self.max_period = max_period
        self.names = [f'ht_dcperiod_{source}']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(min_period, max_period, shape=(1,))
            })

    def process(self, frames, state):
        if len(frames) < self.max_period:
            period = self.min_period
        else:
            window = frames[-self.max_period:]
            prices = np.array([getattr(f, self.source) for f in window])

            analytic_signal = signal.hilbert(prices - prices.mean())
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)

            if instantaneous_frequency[-1] > 0:
                period = min(max(1.0 / instantaneous_frequency[-1], self.min_period), self.max_period)
            else:
                period = self.max_period

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], period)
        if self.write_to_state:
            state[self.names[0]] = period

    def __repr__(self):
        return f"HT_DCPERIOD(source={self.source}, write_to={self.write_to})"
```

**Key points:**
- Requires scipy/numpy
- Computationally expensive
- Needs large window for accuracy

---

## V. Testing Methodology

### Unit Test Template

```python
import unittest
from intraday.features.your_indicator import YourIndicator
from intraday.frame import Frame

class TestYourIndicator(unittest.TestCase):
    def setUp(self):
        self.indicator = YourIndicator(period=14, source='close')

    def test_initialization(self):
        self.assertEqual(self.indicator.period, 14)
        self.assertIn('your_indicator_14_close', self.indicator.names)

    def test_reset(self):
        self.indicator.reset()
        self.assertIsNone(self.indicator.internal_state)

    def test_single_frame(self):
        frames = [Frame(open=100, high=105, low=95, close=102, volume=1000)]
        state = {}
        self.indicator.process(frames, state)
        self.assertIn(self.indicator.names[0], state)

    def test_warmup_period(self):
        frames = [
            Frame(open=100+i, high=105+i, low=95+i, close=100+i, volume=1000)
            for i in range(5)
        ]
        state = {}
        self.indicator.process(frames, state)
        self.assertIsNotNone(state[self.indicator.names[0]])

    def test_known_values(self):
        frames = [...]  # Create known dataset
        state = {}
        self.indicator.process(frames, state)
        expected = ...  # Calculate expected value
        self.assertAlmostEqual(state[self.indicator.names[0]], expected, places=2)

if __name__ == '__main__':
    unittest.main()
```

### Validation Against TA-Lib

```python
import talib
import numpy as np
from intraday.features.your_indicator import YourIndicator
from intraday.frame import Frame

def test_against_talib():
    closes = np.random.randn(100) + 100

    talib_result = talib.RSI(closes, timeperiod=14)

    frames = [
        Frame(open=c, high=c+1, low=c-1, close=c, volume=1000)
        for c in closes
    ]

    indicator = YourIndicator(period=14, source='close')
    indicator.reset()

    our_results = []
    for i in range(len(frames)):
        state = {}
        indicator.process(frames[:i+1], state)
        our_results.append(state.get(indicator.names[0], np.nan))

    our_results = np.array(our_results)

    diff = np.abs(talib_result - our_results)
    max_diff = np.nanmax(diff)

    print(f"Max difference: {max_diff}")
    assert max_diff < 0.01, "Implementation differs from TA-Lib"
```

---

## VI. Implementation Checklist

For each new indicator:

- [ ] Determine pattern type (A-F above)
- [ ] Choose appropriate base class (Feature or TradesFeature)
- [ ] Define `__init__` with proper parameters
- [ ] Set up self.names following naming convention
- [ ] Define gym.spaces with correct bounds
- [ ] Implement reset() to clear state
- [ ] Implement process() with proper logic
- [ ] Handle warmup period (return 0.0 or check len(frames))
- [ ] Implement write_to_frame/state logic
- [ ] Add __repr__ method
- [ ] Write unit tests
- [ ] Validate against TA-Lib (if possible)
- [ ] Add to features/__init__.py
- [ ] Document in FEATURES_ANALYSIS.md

---

## VII. Common Pitfalls

1. **Forgetting to check len(frames) before accessing frames[-2]**
   ```python
   # BAD
   prev_close = frames[-2].close

   # GOOD
   if len(frames) < 2:
       return default_value
   prev_close = frames[-2].close
   ```

2. **Not initializing state variables in reset()**
   ```python
   def reset(self):
       self.ema_value = None  # Don't forget this!
   ```

3. **Incorrect EMA formula during warmup**
   ```python
   # Use SMA for first N values, then switch to EMA
   if self._n_iter < self.period:
       # Simple average
   else:
       # EMA formula
   ```

4. **Missing write_to_state when write_to_frame is set**
   ```python
   # Both branches needed
   if self.write_to_frame:
       setattr(frames[-1], name, value)
   if self.write_to_state:
       state[name] = value
   ```

5. **Division by zero**
   ```python
   if denominator == 0:
       result = 0.0
   else:
       result = numerator / denominator
   ```

---

## VIII. Priority Implementation Order

Based on TA-Lib gap analysis:

### High Priority (Most Used)
1. RSI (Momentum)
2. MACD (Momentum)
3. Bollinger Bands (Overlap)
4. ADX/DMI system (Momentum)
5. Williams %R (Momentum)

### Medium Priority (Common)
6. CCI (Momentum)
7. Aroon (Momentum)
8. ROC variants (Momentum)
9. MFI (Momentum)
10. Standard Deviation (Statistics)

### Low Priority (Specialized)
11. Candlestick patterns (61 total)
12. Hilbert Transform indicators (5 total)
13. Linear Regression suite (Statistics)

---

## IX. File Structure

```
intraday/features/
├── __init__.py           # Export all features
├── rsi.py                # RSI implementation
├── macd.py               # MACD implementation
├── bollinger_bands.py    # Bollinger Bands
├── adx.py                # ADX/DMI system
├── cci.py                # CCI
├── patterns/             # Candlestick patterns
│   ├── __init__.py
│   ├── doji.py
│   ├── engulfing.py
│   └── ...
└── hilbert/              # Hilbert Transform indicators
    ├── __init__.py
    ├── ht_dcperiod.py
    └── ...
```

---

## X. Integration Example

```python
from intraday.features import RSI, MACD, BollingerBands, ATR
from intraday.env import SingleAgentEnv
from intraday.providers import BinanceArchiveProvider
from intraday.processor import IntervalProcessor
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward

provider = BinanceArchiveProvider(
    data_dir='.',
    symbol='BTCUSDT',
    date_from=date(2023, 1, 1),
    date_to=date(2023, 1, 31)
)

processor = IntervalProcessor(method='time', interval=5*60)

features_pipeline = [
    RSI(period=14, source='close', write_to='both'),
    MACD(fast_period=12, slow_period=26, signal_period=9, source='close'),
    BollingerBands(period=20, num_std=2.0, source='close'),
    ATR(period=14, write_to='both'),
]

env = SingleAgentEnv(
    provider=provider,
    processor=processor,
    features_pipeline=features_pipeline,
    action_scheme=BuySellCloseAction(),
    reward_scheme=BalanceReward(),
    initial_balance=10000,
    warm_up_time=timedelta(hours=1)
)

state = env.reset()
print("State keys:", state.keys())
# Output: dict_keys(['rsi_14_close', 'macd_12_26_close', 'macd_signal_9_close',
#                    'macd_hist_12_26_9_close', 'bb_upper_20_close', ...])
```

---

## Summary

**Implementation Flow:**
1. Choose pattern type (A-F)
2. Implement class inheriting from Feature or TradesFeature
3. Define __init__, reset(), process()
4. Handle warmup, write_to logic, naming
5. Test against known values and TA-Lib
6. Document and integrate

**Key Principles:**
- State isolation (reset() must clear everything)
- Warmup awareness (return sensible defaults)
- Standardized naming and spaces
- Consistent write_to logic
- Error handling for edge cases

This methodology ensures all TA-Lib indicators can be systematically implemented with consistency and correctness.
