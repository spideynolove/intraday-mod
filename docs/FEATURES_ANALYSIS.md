# Intraday Features Analysis

Analysis of all 36 feature implementations in `intraday/features/`.

## Summary

**Implementation**: All features are implemented from scratch using custom algorithms. No external technical analysis libraries (like TA-Lib) are used.

**Data Types**: Features operate on two data types:
- **Frame Features** (`Feature` subclass): Process aggregated OHLCV frames
- **Trade Features** (`TradesFeature` subclass): Process raw trade streams

**Look-ahead Bias**: All features properly avoid look-ahead bias. Features with centered windows or lookback operate on historical data with intentional delays.

---

## Feature Catalog

### Trade Stream Features

These inherit from `TradesFeature` and process raw trade data:

#### 1. **AbnormalTrades** (abnormal_trades.py)
- **Purpose**: Detect unusually large trades relative to historical average
- **Method**: EMA of trade amounts, flags trades exceeding `threshold_factor * ema_trade_amount`
- **Data**: Individual trades (buy/sell operations)
- **Look-ahead**: ✅ Safe - uses only past trade history for EMA calculation
- **Output**: Count of abnormal buy trades, count of abnormal sell trades per frame

#### 2. **AverageTrade** (average_trade.py)
- **Purpose**: Track average characteristics of trades over time
- **Method**: EMA calculations for tick direction, spread, trade amount, buy/sell amounts
- **Data**: Trade stream with buy/sell markers
- **Look-ahead**: ✅ Safe - all EMAs use only historical data
- **Output**: ema_trade_tick, ema_trade_spread, ema_trade_amount, ema_buy_amount, ema_sell_amount

#### 3. **PriceDynamics** (price_dynamics.py)
- **Purpose**: Analyze how buy/sell orders move price
- **Method**: Tracks price movement direction per trade type, calculates "ease of movement" metrics
- **Data**: Trade stream with buy/sell operations
- **Look-ahead**: ✅ Safe - compares consecutive trades sequentially
- **Output**: 14 features tracking buy/sell price movements, volumes, and ease metrics
- **Note**: Resets counters after each frame (prevents accumulation bias)

---

### Simple Transformations

Basic mathematical operations on frame fields:

#### 4. **Abs** (abs.py)
- **Purpose**: Absolute value transformation
- **Data**: Any frame field
- **Look-ahead**: ✅ Safe - operates on current frame only
- **Implementation**: `abs(value)`

#### 5. **Clip** (clip.py)
- **Purpose**: Clamp values to specified range
- **Data**: Any frame field
- **Look-ahead**: ✅ Safe - operates on current frame only
- **Implementation**: `min(max(value, minval), maxval)`

#### 6. **Copy** (copy.py)
- **Purpose**: Copy frame fields to state (utility feature)
- **Data**: Frame fields
- **Look-ahead**: ✅ Safe - direct copy
- **Implementation**: Simple attribute copy

#### 7. **Delta** (delta.py)
- **Purpose**: Difference between two frame fields
- **Data**: Any two frame fields (e.g., close - open)
- **Look-ahead**: ✅ Safe - operates on current frame only
- **Implementation**: `value1 - value2`

#### 8. **Div** (div.py)
- **Purpose**: Division of frame fields
- **Data**: Any two frame fields or field/constant
- **Look-ahead**: ✅ Safe - operates on current frame only
- **Implementation**: `value1 / value2` (handles division by zero)

#### 9. **DivDelta** (div_delta.py)
- **Purpose**: Normalized difference: `(value1 - value2) / value3`
- **Data**: Three frame fields
- **Look-ahead**: ✅ Safe - operates on current frame only
- **Implementation**: `(a - b) / c`

#### 10. **Mul** (mul.py)
- **Purpose**: Multiplication of frame fields
- **Data**: Any two frame fields or field/constant
- **Look-ahead**: ✅ Safe - operates on current frame only
- **Implementation**: `value1 * value2`

#### 11. **Log** (log.py)
- **Purpose**: Natural logarithm with sign preservation
- **Data**: Any frame field
- **Look-ahead**: ✅ Safe - operates on current frame only
- **Implementation**: `sign(value) * log(abs(value))` if `|value| > 1`, else 0

---

### Price Transformations

Features that encode or transform price data:

#### 12. **Change** (change.py)
- **Purpose**: Price change over N periods
- **Data**: Any frame field (typically close, vwap)
- **Look-ahead**: ✅ Safe - compares current frame to frame N periods ago
- **Implementation**: `current_value - past_value`
- **Period**: Configurable

#### 13. **LogDelta** (log_delta.py)
- **Purpose**: Log of price difference
- **Data**: Two frame fields (e.g., close, open)
- **Look-ahead**: ✅ Safe - operates on current frame
- **Implementation**: `sign(delta) * log(|delta|)` if `|delta| >= 1`, else 0

#### 14. **LogReturn** (log_return.py)
- **Purpose**: Logarithmic return calculation
- **Data**: Two positive frame fields
- **Look-ahead**: ✅ Safe - operates on current frame
- **Implementation**: `log(value1 / value2)`
- **Note**: Standard in quantitative finance

#### 15. **Return** (return_feature.py)
- **Purpose**: Simple return calculation
- **Data**: Two frame fields
- **Look-ahead**: ✅ Safe - operates on current frame
- **Implementation**: `value1 / value2 - 1.0`

#### 16. **SemiLogReturn** (semi_log_return.py)
- **Purpose**: Hybrid return metric (parabolic for small returns, log for large)
- **Data**: Two positive frame fields
- **Look-ahead**: ✅ Safe - operates on current frame
- **Implementation**: Parabolic when ratio < 1, logarithmic when ratio >= 1
- **Use Case**: Reduces impact of small fluctuations

#### 17. **PriceEncoder** (price_encoder.py)
- **Purpose**: Flexible price encoding with multiple methods
- **Data**: Frame fields (OHLC)
- **Look-ahead**: ✅ Safe - compares to N periods ago
- **Methods**: raw, delta, return, logreturn
- **Period**: Configurable
- **Base**: Optional reference price (can be different field)

---

### Time Features

#### 18. **TimeEncoder** (time_encoder.py)
- **Purpose**: Encode temporal information
- **Data**: Frame timestamps
- **Look-ahead**: ✅ Safe - encodes current time only
- **Output**:
  - `yday`: Day of year [0,1]
  - `wday`: Day of week [0,1]
  - `time`: Time of day [0,1]
- **Use Case**: Capture seasonality, weekly patterns, intraday cycles

---

### Moving Averages & Smoothing

#### 19. **EMA** (ema.py)
- **Purpose**: Exponential Moving Average
- **Data**: Any frame field(s)
- **Look-ahead**: ✅ Safe - uses only current and past frames
- **Implementation**: Standard EMA formula with warmup period
- **Period**: Configurable
- **Note**: Uses simple average for first N samples, then switches to EMA

#### 20. **KAMA** (kama.py)
- **Purpose**: Kaufman's Adaptive Moving Average
- **Data**: Frame field (typically close)
- **Look-ahead**: ✅ Safe - uses efficiency ratio from past periods
- **Implementation**: Custom from scratch
- **Parameters**: (fast_period, slow_period, kama_period)
- **Features**: Adapts smoothing factor based on market efficiency

#### 21. **ZEMA** (zema.py)
- **Purpose**: Zero-lag Exponential Moving Average
- **Data**: Frame field (typically hlc3)
- **Look-ahead**: ✅ Safe - uses lag compensation from past periods
- **Implementation**: `ema(value + K * (value - value_lagged))`
- **Parameters**: period, alpha, K
- **Use Case**: Reduces lag while maintaining smoothness

#### 22. **GaussianSmooth** (gaussian_smooth.py)
- **Purpose**: Gaussian kernel smoothing
- **Data**: Frame field (typically vwap)
- **Look-ahead**: ✅ Safe - centered smoothing with intentional delay
- **Implementation**: Applies Gaussian kernel centered on frame N-radius-1, uses frames [N-2R-1:N]
- **Period**: 2*radius + 1
- **Write**: Only to "frame" (not state) - smoothed value written to delayed frame position
- **Delay**: Output appears radius frames after input
- **Use Case**: Noise reduction; note the inherent lag

#### 23. **WMASignal** (wma_signal.py)
- **Purpose**: Hull-like moving average signal
- **Data**: Frame field (typically hlc3)
- **Look-ahead**: ✅ Safe - uses only past frames
- **Implementation**: WMA of WMA, with prediction and signal calculation
- **Period**: Configurable
- **Output**: Signal strength based on predicted vs smoothed predicted price

---

### Momentum & Oscillators

#### 24. **EfficiencyRatio** (efficiency_ratio.py)
- **Purpose**: Measure price movement efficiency (Kaufman's ER)
- **Data**: Frame field (typically close)
- **Look-ahead**: ✅ Safe - uses only past N frames
- **Implementation**: `net_change / sum_of_absolute_changes`
- **Range**: [0, 1] where 1 = perfectly efficient (straight line)
- **Period**: Configurable

#### 25. **Stochastic** (stochastic.py)
- **Purpose**: Stochastic oscillator
- **Data**: (low, high, close)
- **Look-ahead**: ✅ Safe - uses highest/lowest over past N periods
- **Implementation**: `(close - lowest) / (highest - lowest) - 0.5`
- **Range**: [-0.5, 0.5]
- **Period**: Configurable

---

### Volume-Based Indicators

#### 26. **ADL** (adl.py)
- **Purpose**: Accumulation/Distribution Line
- **Data**: (low, high, close, volume)
- **Look-ahead**: ✅ Safe - cumulative calculation
- **Implementation**: Cumulative money flow multiplier * volume
- **Formula**: `((close - low) - (high - close)) / (high - low) * volume`
- **Type**: Cumulative (unbounded)

#### 27. **CMF** (cmf.py)
- **Purpose**: Chaikin Money Flow
- **Data**: (low, high, close, volume)
- **Look-ahead**: ✅ Safe - uses past N periods
- **Implementation**: Sum of money flow volumes / sum of volumes over period
- **Range**: [-1, 1]
- **Period**: Configurable

#### 28. **OBV** (obv.py)
- **Purpose**: On-Balance Volume
- **Data**: (close, volume)
- **Look-ahead**: ✅ Safe - cumulative based on price direction
- **Implementation**: Add volume if close > prev_close, subtract if close < prev_close
- **Type**: Cumulative (unbounded)

#### 29. **EOM** (eom.py)
- **Purpose**: Ease of Movement
- **Data**: (low, high, volume)
- **Look-ahead**: ✅ Safe - compares current to previous frame
- **Implementation**: `(distance * box_ratio) / (volume * price_factor^2)`
- **Optional**: price_factor, volume_factor for normalization

#### 30. **VI** (vi.py)
- **Purpose**: Positive/Negative Volume Index
- **Data**: (close, volume)
- **Look-ahead**: ✅ Safe - updates based on volume direction
- **Implementation**: Updates PVI when volume increases, NVI when volume decreases
- **Type**: Cumulative indices starting at 100

---

### Pattern Detection

#### 31. **Fractal** (fractal.py)
- **Purpose**: Williams Fractal detection + support/resistance levels
- **Data**: (low, high) or single field
- **Look-ahead**: ✅ Safe - centered detection with intentional delay
- **Implementation**: Detects local maxima/minima within radius*2+1 window
- **Fractal Detection**: Analyzes frame at position N-radius-1 using frames [N-2R-1:N]
- **Support/Resistance**: Counts detected fractals within threshold of current price
- **Period**: 2*radius + 1
- **Delay**: Fractal markers appear radius frames after occurrence
- **Output**: Fractal markers on delayed frames, real-time support/resistance counts
- **Use Case**: Support/resistance levels with delayed fractal confirmation

#### 32. **CumulativeSum** (cumulative_sum.py)
- **Purpose**: CUSUM change detection
- **Data**: Frame field (typically close)
- **Look-ahead**: ✅ Safe - sequential monitoring of deviations
- **Implementation**: Tracks positive/negative cumulative deviations from expected value
- **Output**: Signal when deviation exceeds threshold (+1 or -1)
- **Threshold**: Can be constant or dynamic (another frame field)
- **Use Case**: Regime change detection

---

### Market Microstructure

#### 33. **FractalDimension** (fractal_dimension.py)
- **Purpose**: Measure market complexity/choppiness
- **Data**: (low, high)
- **Look-ahead**: ✅ Safe - uses past N periods
- **Implementation**: Hurst exponent estimation via range analysis
- **Formula**: `log(N1+N2) - log(N3)) / log(2)` where Ni are normalized ranges
- **Range**: Typically [0, 2] where higher = more complex/random
- **Period**: Configurable

#### 34. **MarketDimension** (market_dimension.py)
- **Purpose**: Measure consolidation vs trending (0=consolidated, 1=trending)
- **Data**: (low, high)
- **Look-ahead**: ✅ Safe - uses past N periods
- **Implementation**: `(SN - S1) / (S2 - S1)` where SN=sum of ranges, S1=period range
- **Range**: [0, 1]
- **Period**: Configurable

---

### Advanced Indicators

#### 35. **HeikenAshi** (heiken_ashi.py)
- **Purpose**: Heiken-Ashi candlestick transformation
- **Data**: (open, high, low, close)
- **Look-ahead**: ✅ Safe - uses only current and previous frame
- **Implementation**: Standard HA formulas
- **Output**: HA open, high, low, close
- **Use Case**: Noise reduction, trend clarity

#### 36. **ParabolicSAR** (parabolic_sar.py)
- **Purpose**: Parabolic Stop and Reverse
- **Data**: (low, high)
- **Look-ahead**: ✅ Safe - uses 2-3 frame lookback
- **Implementation**: Dual SAR tracking (upward and downward)
- **Lookback**: Checks frames[-2] and frames[-3] to prevent SAR inside prior candles
- **Parameters**: acceleration, max_velocity
- **Output**: u_sar, u_reset, d_sar, d_reset
- **Use Case**: Trailing stop levels

#### 37. **Snapshot** (snapshot.py)
- **Purpose**: Multi-dimensional price history representation
- **Data**: (close, high, low, volume, true_range)
- **Look-ahead**: ✅ Safe - maintains sliding window of past N frames
- **Implementation**: Normalized arrays of price distance, proximity, IOU, volume, TR
- **Output**: 5 arrays of length=period
- **Period**: Configurable
- **Use Case**: Deep learning input (convolutional features)

---

## Look-ahead Bias Assessment

**Result**: ✅ All 36 features avoid look-ahead bias.

**Key Insight**: At time T with frames [0...T] available, all features use only data from times ≤T. Features that process delayed frames (GaussianSmooth, Fractal) or use lookback (ParabolicSAR) still only access historical data.

**Centered Features**: GaussianSmooth and Fractal process frame at position N-R-1 using frames [N-2R-1:N]. While this creates an R-frame delay, all input frames are historical relative to the current timestep N. This is intentional smoothing lag, not data leakage.

---

## System Integration

**Warmup Handling**: The environment (`env.py`) provides automatic warmup before agent interaction:

1. **Configuration** (env.py:48):
   ```python
   warm_up_time: Optional[Union[Real, timedelta]] = 10 * 60  # 10 minutes default
   ```

2. **Episode Adjustment** (env.py:236-251):
   - Subtracts `warm_up_time` from `episode_start_datetime`
   - Provider starts delivering data earlier than episode start

3. **Feature Initialization** (env.py:333-357):
   - Processes trades before `episode_start_datetime`
   - Features accumulate history during warmup
   - Agent receives first state only after `frame.time_end >= episode_start_datetime`

**Result**: Features have sufficient history before agent sees any observations. No manual warmup tracking needed in feature implementations.

**Warmup Requirements by Feature Type**:
- **Period-based** (EMA, KAMA, Stochastic): Need `period` frames for stable output
- **Centered** (GaussianSmooth, Fractal): Need `radius` frames before output begins
- **Lookback** (ParabolicSAR, HeikenAshi): Need 2-3 frames minimum
- **Cumulative** (ADL, OBV, VI): Functional from frame 0, stabilize over 50-100 frames
- **Trade-based** (AverageTrade): Need `trades_period` trades for EMA stability

**Edge Case Handling**: Each feature handles insufficient data gracefully:
- GaussianSmooth: Returns early if `i < 0`
- Fractal: No output until `N > radius`
- ParabolicSAR: Checks `len(frames) > 1` and `len(frames) > 2`
- Period-based: Use simple average until period reached

---

## Implementation Quality

### Strengths

1. **All from Scratch**: No external dependencies on TA libraries
2. **Consistent Interface**: All inherit from `Feature` or `TradesFeature` base classes
3. **Configurable Output**: `write_to` parameter controls where features are stored
4. **Type Safety**: Proper type checking and validation
5. **Gymnasium Integration**: Proper space definitions for RL environments
6. **Reset Support**: All features implement reset() for episode boundaries
7. **Numerical Stability**: Division by zero checks, epsilon values for stability

### Areas for Improvement

1. **Missing Unit Tests**: As noted in legacy/README.md, not all features have tests
2. **Limited Documentation**: No docstrings per code standards (intentional)
3. **Numerical Precision**: Some features could benefit from Kahan summation for long-running cumulative calculations (ADL, OBV)

---

## Usage Recommendations

### For Live Trading
- ✅ All features are safe for live trading
- Set `warm_up_time` appropriately based on slowest feature (typically 10-30 minutes)
- Note: GaussianSmooth and Fractal have intentional R-frame delay

### For Backtesting
- ✅ All features work correctly with proper `warm_up_time`
- Recommend `warm_up_time` ≥ max(feature_periods) * frame_duration

### For Deep Learning
- Snapshot: Designed for CNN inputs
- TimeEncoder: Captures temporal patterns
- Combination of price transformations (LogReturn) + technical indicators recommended

### For Traditional ML
- Price transformations (Return, LogReturn) for stationarity
- Efficiency ratios and momentum indicators for regime detection
- Volume indicators for market participation signals

---

## Feature Categories by Data Dependency

**Level 0 - Current Frame Only**: Abs, Clip, Copy, Delta, Div, DivDelta, Mul, Log, LogDelta, LogReturn, Return, SemiLogReturn, TimeEncoder

**Level 1 - Previous Frame**: HeikenAshi, EOM, OBV, VI, Change (period=2)

**Level 2 - Trade Stream**: AbnormalTrades, AverageTrade, PriceDynamics

**Level 3 - Sliding Window**: EMA, KAMA, ZEMA, WMASignal, EfficiencyRatio, Stochastic, ADL, CMF, CumulativeSum, FractalDimension, MarketDimension, Snapshot, PriceEncoder

**Level 4 - Centered/Lookback**: GaussianSmooth (delayed), Fractal (delayed), ParabolicSAR (2-3 frame lookback)

---

## Conclusion

The feature library is professionally implemented with **zero look-ahead bias** across all 36 features. Features with centered windows (GaussianSmooth, Fractal) and lookback (ParabolicSAR) operate on historical data with intentional delays that are inherent to their algorithms.

The environment's `warm_up_time` mechanism provides automatic initialization, allowing features to accumulate sufficient history before agent interaction. Each feature handles edge cases gracefully, making the system robust for both research and production use.

All implementations are custom-built from scratch, demonstrating deep understanding of technical analysis, market microstructure, and proper time-series handling in financial ML applications.
