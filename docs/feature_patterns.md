# Feature Implementation Pattern Analysis

## Pattern Groups Identified

### 1. STATELESS UNARY TRANSFORMERS (7 features)
**Common Pattern:**
- Single source field(s) -> apply function -> output
- No internal state
- Pure transformation on last_frame

**Features:** Abs, Log, Copy, Clip, Change, Semi_Log_Return, Log_Return

**Common Code:**
```python
def __init__(self, source: Union[str, Sequence[str]], write_to):
    super().__init__(write_to=write_to)
    self.source = [source] if isinstance(source, str) else source
    self.names = [f"{operation}_{name}" for name in self.source]
    self.spaces = OrderedDict({name: gym.spaces.Box(...) for name in self.names})

def process(self, frames, state):
    last_frame = frames[-1]
    for i, name in enumerate(self.source):
        value = TRANSFORM(getattr(last_frame, name))
        if self.write_to_frame: setattr(last_frame, self.names[i], value)
        if self.write_to_state: state[self.names[i]] = value
```

### 2. STATELESS BINARY TRANSFORMERS (6 features)
**Common Pattern:**
- Two source fields (or field + constant) -> apply operation -> output
- No internal state

**Features:** Delta, Div, Mul, Div_Delta, Log_Delta, Return_Feature

**Common Code:**
```python
def __init__(self, source: Union[Tuple, Sequence[Tuple]], write_to):
    super().__init__(write_to=write_to)
    self.source = [source] if isinstance(source, Tuple) else source
    self.names = [f"{op}_{name1}_{name2}" for name1, name2 in self.source]
    self.spaces = OrderedDict({name: gym.spaces.Box(...) for name in self.names})

def process(self, frames, state):
    last_frame = frames[-1]
    for i, (name1, name2) in enumerate(self.source):
        value1 = getattr(last_frame, name1)
        value2 = getattr(last_frame, name2) if isinstance(name2, str) else name2
        result = OPERATION(value1, value2)
        if self.write_to_frame: setattr(last_frame, self.names[i], result)
        if self.write_to_state: state[self.names[i]] = result
```

### 3. STATEFUL EXPONENTIAL MOVING AVERAGES (3 features)
**Common Pattern:**
- Maintains EMA state
- First value initializes, subsequent values use EMA formula
- _update_average_value() method

**Features:** EMA, ZEMA, WMA_Signal

**Common Code:**
```python
def __init__(self, period, source, write_to):
    super().__init__(write_to=write_to, period=period)
    self.source = source
    self.ema_value = None  # or specific state variables
    self._n_iter = 0
    self._ema_factor = 2 / (period + 1)

def reset(self):
    self.ema_value = None
    self._n_iter = 0

def process(self, frames, state):
    value = getattr(frames[-1], self.source)
    if self.ema_value is None:
        self.ema_value = value
    elif self._n_iter < self.period:
        self.ema_value = (self.ema_value * self._n_iter + value) / (self._n_iter + 1)
    else:
        self.ema_value = self.ema_value * (1 - self._ema_factor) + value * self._ema_factor
    self._n_iter += 1
    # write_to_frame/state logic
```

### 4. STATEFUL ADAPTIVE AVERAGES (2 features)
**Common Pattern:**
- Adaptive smoothing based on market conditions
- Uses helper features/calculations

**Features:** KAMA, Efficiency_Ratio

### 5. CUMULATIVE ACCUMULATORS (4 features)
**Common Pattern:**
- Accumulates values over frames
- reset() clears accumulator

**Features:** Cumulative_Sum, ADL, OBV, CMF

**Common Code:**
```python
def __init__(self, source, write_to):
    super().__init__(write_to=write_to)
    self.cumulative_value = 0

def reset(self):
    self.cumulative_value = 0

def process(self, frames, state):
    increment = CALCULATE_INCREMENT(frames[-1])
    self.cumulative_value += increment
    # write_to_frame/state
```

### 6. ROLLING WINDOW INDICATORS (5 features)
**Common Pattern:**
- Requires multiple frames (sliding window)
- Calculates over frames[-period:]

**Features:** Stochastic, VI, Gaussian_Smooth, Fractal, Parabolic_SAR

### 7. TRADE-STREAM FEATURES (3 features - TradesFeature subclass)
**Common Pattern:**
- Implements update(trades) method
- Accumulates trade-level data
- process() outputs aggregated results

**Features:** AverageTrade, AbnormalTrades, PriceDynamics

**Common Code:**
```python
def __init__(self, trades_period, write_to):
    super().__init__(trades_period=trades_period, write_to=write_to)
    self._ema_factor = 2 / (trades_period + 1)
    self._n_iter = 0
    # state variables

def reset(self):
    self._n_iter = 0
    # reset state

def update(self, trades):
    trade = trades[-1]
    # accumulate trade data

def process(self, frames, state):
    # output accumulated results
    # reset accumulators
```

### 8. SPECIALIZED ENCODERS (4 features)
**Common Pattern:**
- Domain-specific transformations
- Often stateful

**Features:** Price_Encoder, Time_Encoder, Snapshot, Heiken_Ashi

### 9. SPECIALIZED INDICATORS (2 features)
**Common Pattern:**
- Unique algorithms
- Don't fit other patterns

**Features:** EOM, Fractal_Dimension, Market_Dimension
