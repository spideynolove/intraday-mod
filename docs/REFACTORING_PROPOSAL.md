# Feature Refactoring Proposal: Base Class Abstraction

## Executive Summary

**Feasibility:** YES - 28 of 36 features (78%) can be generalized into 4 new base classes.

**Benefits:**
- Eliminate ~400-600 lines of duplicated boilerplate code
- Standardize write_to_frame/state logic (currently duplicated 36 times)
- Simplify new feature creation (inherit + implement one method)
- Maintain backward compatibility

**Trade-offs:**
- Slightly increased abstraction complexity
- Refactoring effort for 28 files

---

## Current Architecture

```
Feature (ABC)
├── TradesFeature (ABC) - 3 features
└── Individual Features - 33 features (all implement same boilerplate)
```

**Problem:** Every feature reimplements:
1. `__init__` with write_to handling
2. names list creation
3. spaces OrderedDict creation
4. write_to_frame/state conditional logic in process()
5. `__repr__` implementation

---

## Proposed Architecture

```
Feature (ABC)
├── UnaryTransformer (new base) - 7 features
├── BinaryTransformer (new base) - 6 features
├── StatefulEMA (new base) - 3 features
├── CumulativeAccumulator (new base) - 4 features
├── TradesFeature (existing) - 3 features
└── Specialized Features - 13 features (unchanged)
```

---

## Detailed Refactoring Groups

### GROUP 1: UnaryTransformer Base Class (7 features)

**Applicable Features:**
- Abs, Log, Copy, Clip, Change, Semi_Log_Return, Log_Return

**Current Duplication:** Each feature has ~35-45 lines of identical code.

**Proposed Base Class:**

```python
from abc import abstractmethod
from typing import Union, Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature

class UnaryTransformer(Feature):
    def __init__(
        self,
        source: Union[str, Sequence[str]],
        operation_name: str,
        write_to: Literal["state", "frame", "both"] = "state",
        box_low: float = -math.inf,
        box_high: float = math.inf,
    ):
        super().__init__(write_to=write_to)
        self.source = [source] if isinstance(source, str) else list(source)
        self.operation_name = operation_name
        self.names = [f"{operation_name}_{name}" for name in self.source]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                name: gym.spaces.Box(box_low, box_high, shape=(1,))
                for name in self.names
            })
        else:
            self.spaces = OrderedDict()

    @abstractmethod
    def transform(self, value):
        raise NotImplementedError()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            value = getattr(last_frame, name)
            result = self.transform(value)
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result

    def __repr__(self):
        return f"{self.__class__.__name__}(source={self.source}, write_to={self.write_to})"
```

**Example Migration - Abs:**

Before (46 lines):
```python
class Abs(Feature):
    def __init__(self, source, write_to):
        super().__init__(write_to=write_to)
        # ... 30 lines of boilerplate

    def process(self, frames, state):
        # ... 8 lines of write_to logic
```

After (5 lines):
```python
class Abs(UnaryTransformer):
    def __init__(self, source, write_to="state"):
        super().__init__(source, "abs", write_to, box_low=0, box_high=math.inf)

    def transform(self, value):
        return abs(value)
```

**Code Reduction:** 46 lines → 5 lines per feature (×7 features = ~287 lines saved)

---

### GROUP 2: BinaryTransformer Base Class (6 features)

**Applicable Features:**
- Delta, Div, Mul, Div_Delta, Log_Delta, Return_Feature

**Proposed Base Class:**

```python
class BinaryTransformer(Feature):
    def __init__(
        self,
        source: Union[Tuple[str, Union[str, Real]], Sequence[Tuple[str, Union[str, Real]]]],
        operation_name: str,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to)
        if isinstance(source, Tuple) and len(source) == 2:
            self.source = [source]
        else:
            self.source = list(source)
        for name1, name2 in self.source:
            assert isinstance(name1, str) and isinstance(name2, (str, Real))
        self.operation_name = operation_name
        self.names = [
            f"{operation_name}_{name1}_{str(name2).replace('.', '_')}"
            for name1, name2 in self.source
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                for name in self.names
            })
        else:
            self.spaces = OrderedDict()

    @abstractmethod
    def transform(self, value1, value2):
        raise NotImplementedError()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, (name1, name2) in enumerate(self.source):
            value1 = getattr(last_frame, name1)
            value2 = getattr(last_frame, name2) if isinstance(name2, str) else name2
            result = self.transform(value1, value2)
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result

    def __repr__(self):
        return f"{self.__class__.__name__}(source={self.source}, write_to={self.write_to})"
```

**Example Migration - Delta:**

Before (58 lines):
```python
class Delta(Feature):
    def __init__(self, source, write_to):
        # ... 40 lines of boilerplate

    def process(self, frames, state):
        # ... error checking, write_to logic
```

After (5 lines):
```python
class Delta(BinaryTransformer):
    def __init__(self, source=(("close", "open"),), write_to="state"):
        super().__init__(source, "delta", write_to)

    def transform(self, value1, value2):
        if not isinstance(value1, Real) or not isinstance(value2, Real):
            return 0.0
        return value1 - value2
```

**Code Reduction:** 58 lines → 7 lines per feature (×6 features = ~306 lines saved)

---

### GROUP 3: StatefulEMA Base Class (3 features)

**Applicable Features:**
- EMA (multi-source support)
- AverageTrade (TradesFeature with EMA logic)
- AbnormalTrades (TradesFeature with EMA logic)

**Proposed Base Class:**

```python
class StatefulEMA(Feature):
    def __init__(
        self,
        period: int,
        source: Union[str, Sequence[str]],
        operation_name: str,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = [source] if isinstance(source, str) else list(source)
        self.operation_name = operation_name
        self.names = [f"ema_{period}_{name}" for name in self.source]
        self._ema_factor = 2 / (period + 1)
        self._n_iter = 0

        for name in self.source:
            setattr(self, f"_state_{name}", None)

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                for name in self.names
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._ema_factor = 2 / (self.period + 1)
        self._n_iter = 0
        for name in self.source:
            setattr(self, f"_state_{name}", None)

    @abstractmethod
    def extract_value(self, frames: Sequence[Frame], source_name: str):
        raise NotImplementedError()

    def _update_ema(self, state_name: str, new_value):
        if new_value is None:
            return None
        old_average = getattr(self, state_name)
        if old_average is None:
            new_average = new_value
        elif self._n_iter < self.period:
            new_average = (old_average * self._n_iter + new_value) / (self._n_iter + 1)
        else:
            new_average = old_average * (1 - self._ema_factor) + new_value * self._ema_factor
        setattr(self, state_name, new_average)
        return new_average

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            new_value = self.extract_value(frames, name)
            new_average = self._update_ema(f"_state_{name}", new_value)
            if self.write_to_frame:
                setattr(last_frame, self.names[i], new_average)
            if self.write_to_state:
                state[self.names[i]] = new_average
        self._n_iter += 1

    def __repr__(self):
        return f"{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})"
```

**Example Migration - EMA:**

Before (76 lines):
```python
class EMA(Feature):
    def __init__(self, period, source, write_to):
        # ... 37 lines

    def reset(self):
        # ... 4 lines

    def process(self, frames, state):
        # ... 10 lines

    def _update_average_value(self, name, new_value):
        # ... 13 lines
```

After (5 lines):
```python
class EMA(StatefulEMA):
    def __init__(self, period, source, write_to="state"):
        super().__init__(period, source, "ema", write_to)

    def extract_value(self, frames, source_name):
        return getattr(frames[-1], source_name)
```

**Code Reduction:** 76 lines → 5 lines (×3 features = ~213 lines saved)

---

### GROUP 4: CumulativeAccumulator Base Class (4 features)

**Applicable Features:**
- Cumulative_Sum, ADL, OBV, CMF

**Proposed Base Class:**

```python
class CumulativeAccumulator(Feature):
    def __init__(
        self,
        output_name: str,
        write_to: Literal["state", "frame", "both"] = "state",
        **kwargs
    ):
        super().__init__(write_to=write_to, **kwargs)
        self.output_name = output_name
        self.names = [output_name]
        self.cumulative_value = 0

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                output_name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self.cumulative_value = 0

    @abstractmethod
    def calculate_increment(self, frames: Sequence[Frame]) -> float:
        raise NotImplementedError()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        increment = self.calculate_increment(frames)
        self.cumulative_value += increment

        if self.write_to_frame:
            setattr(frames[-1], self.output_name, self.cumulative_value)
        if self.write_to_state:
            state[self.output_name] = self.cumulative_value

    def __repr__(self):
        return f"{self.__class__.__name__}(write_to={self.write_to})"
```

**Example Migration - OBV:**

Before (45 lines):
```python
class OBV(Feature):
    def __init__(self, source='close', write_to='both'):
        # ... 25 lines boilerplate

    def reset(self):
        self.obv = 0

    def process(self, frames, state):
        # ... 15 lines
```

After (8 lines):
```python
class OBV(CumulativeAccumulator):
    def __init__(self, source='close', write_to='both'):
        super().__init__('obv', write_to)
        self.source = source

    def calculate_increment(self, frames):
        frame = frames[-1]
        if len(frames) < 2:
            return 0
        prev_close = getattr(frames[-2], self.source)
        curr_close = getattr(frame, self.source)
        if curr_close > prev_close:
            return frame.volume
        elif curr_close < prev_close:
            return -frame.volume
        return 0
```

**Code Reduction:** 45 lines → 12 lines per feature (×4 features = ~132 lines saved)

---

## Total Impact Summary

| Group | Features | Lines Saved | Complexity Reduction |
|-------|----------|-------------|---------------------|
| UnaryTransformer | 7 | ~287 | High |
| BinaryTransformer | 6 | ~306 | High |
| StatefulEMA | 3 | ~213 | Medium |
| CumulativeAccumulator | 4 | ~132 | Medium |
| **TOTAL** | **20** | **~938** | **High** |

**Additional Benefits:**
- 8 Specialized features unchanged (Price_Encoder, Time_Encoder, Heiken_Ashi, EOM, Fractal_Dimension, Market_Dimension, Snapshot, Gaussian_Smooth)
- 5 Rolling Window features unchanged (Stochastic, VI, Fractal, Parabolic_SAR, KAMA)
- 3 TradesFeature implementations could share EMA logic via mixin

---

## Implementation Strategy

### Phase 1: Create Base Classes
1. Add to `intraday/feature.py`:
   - UnaryTransformer
   - BinaryTransformer
   - StatefulEMA
   - CumulativeAccumulator

### Phase 2: Migrate in Order of Safety
1. **Week 1:** UnaryTransformer (7 features) - safest, pure functions
2. **Week 2:** BinaryTransformer (6 features) - simple operations
3. **Week 3:** CumulativeAccumulator (4 features) - stateful but simple
4. **Week 4:** StatefulEMA (3 features) - most complex

### Phase 3: Testing
- Unit tests for each base class
- Regression tests to ensure outputs match original implementations
- Integration tests with actual trading environment

### Phase 4: Documentation
- Update FEATURES_ANALYSIS.md with new architecture
- Add examples for creating new features using base classes
- Document migration from old to new patterns

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Breaking existing code | High | Maintain backward compatibility; deprecate old patterns gradually |
| Performance regression | Medium | Benchmark before/after; base classes should be zero-cost abstractions |
| Increased learning curve | Low | Clear documentation; examples in docstrings |
| Edge case bugs | Medium | Comprehensive unit tests; gradual rollout |

---

## Recommendation

**PROCEED WITH REFACTORING** using phased approach.

**Priority Order:**
1. UnaryTransformer (highest ROI, lowest risk)
2. BinaryTransformer (high ROI, low risk)
3. CumulativeAccumulator (medium ROI, low risk)
4. StatefulEMA (medium ROI, medium risk)

**Expected Outcome:**
- 938+ lines of duplicated code eliminated
- 20 features simplified to 5-12 lines each
- Standardized patterns for future feature development
- Maintained backward compatibility
- Zero performance impact (pure refactoring)

---

## Next Steps

1. Get approval for refactoring approach
2. Create feature branch: `refactor/base-class-abstraction`
3. Implement Phase 1 (base classes) with comprehensive tests
4. Migrate features in phases 2-4
5. Update documentation
6. Merge to main after all tests pass
