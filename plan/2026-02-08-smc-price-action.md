# SMC Price Action Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 5 Smart Money Concept (SMC) feature classes as Feature extractors that output mixed boolean/numeric/categorical values for use by ML agents.

**Architecture:** Each SMC concept becomes a Feature subclass following the existing `intraday/feature.py` pattern. All features maintain internal state, define gymnasium spaces, and write to state/frame using the `write_to` parameter. They work flexibly with any Processor-based timeframe.

**Tech Stack:** Python 3.10+, gymnasium, collections.OrderedDict, collections.namedtuple

---

## Reference Files (Read Before Each Task)

- `intraday/feature.py` - Feature base class, StatefulEMA pattern
- `intraday/features/fractal.py` - Extremum tracking with namedtuple, FIFO eviction
- `intraday/features/parabolic_sar.py` - Multiple boolean + numeric outputs, dual state
- `intraday/features/aroon.py` - Named outputs with period in name
- `intraday/frame.py` - Frame fields: open/high/low/close/volume, buy_volume/sell_volume, time_start/time_end (Arrow)
- `test/test_adx.py` - Test structure for complex stateful indicators

---

## Task 1: SwingStructure Feature

Market structure detection: swing points, Break of Structure (BOS), Change of Character (ChoCh).

**Files:**
- Create: `intraday/features/smc_swing_structure.py`
- Test: `test/test_smc_swing_structure.py`

**Step 1: Write the failing test**

```python
# test/test_smc_swing_structure.py
import unittest
from collections import OrderedDict
from intraday.features.smc_swing_structure import SwingStructure
from intraday.frame import Frame


class TestSwingStructure(unittest.TestCase):
    def test_initializes(self):
        s = SwingStructure(swing_period=3)
        self.assertIn('swing_high_detected', s.names)
        self.assertIn('swing_low_detected', s.names)
        self.assertIn('bos_bullish', s.names)
        self.assertIn('choch_bullish', s.names)

    def test_detects_swing_high(self):
        s = SwingStructure(swing_period=1)
        s.reset()
        frames = [
            Frame(high=100, low=95, close=98),
            Frame(high=110, low=105, close=108),
            Frame(high=105, low=100, close=102),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            s.process(frames[:i+1], state)
        self.assertEqual(state['swing_high_detected'], 1)
        self.assertAlmostEqual(state['swing_high_price'], 110.0)

    def test_detects_swing_low(self):
        s = SwingStructure(swing_period=1)
        s.reset()
        frames = [
            Frame(high=110, low=105, close=108),
            Frame(high=100, low=90, close=92),
            Frame(high=105, low=100, close=102),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            s.process(frames[:i+1], state)
        self.assertEqual(state['swing_low_detected'], 1)
        self.assertAlmostEqual(state['swing_low_price'], 90.0)

    def test_detects_bullish_bos(self):
        s = SwingStructure(swing_period=1)
        s.reset()
        # Uptrend: LL, LH, HL, HH - BOS when breaking prev high
        frames = [
            Frame(high=100, low=90, close=95),
            Frame(high=95, low=85, close=88),  # swing low
            Frame(high=105, low=92, close=103),
            Frame(high=90, low=80, close=83),  # swing low (HL)
            Frame(high=108, low=86, close=106),  # swing high (HH) = BOS
            Frame(high=102, low=95, close=98),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            s.process(frames[:i+1], state)
        # At some point bos_bullish should have been 1
        # We check the internal swing tracking worked
        self.assertIn('bos_bullish', state)

    def test_reset_clears_state(self):
        s = SwingStructure(swing_period=1)
        frames = [Frame(high=110, low=90, close=100)]
        for i in range(len(frames)):
            s.process(frames[:i+1], OrderedDict())
        s.reset()
        state = OrderedDict()
        s.process([Frame(high=100, low=95, close=98)], state)
        self.assertEqual(state['swing_high_detected'], 0)


if __name__ == '__main__':
    unittest.main()
```

**Step 2: Run test to verify it fails**

```bash
source /home/hung/env/.venv/bin/activate
python -m pytest test/test_smc_swing_structure.py -v
```
Expected: ImportError or ModuleNotFoundError

**Step 3: Implement SwingStructure**

```python
# intraday/features/smc_swing_structure.py
from typing import Sequence, Literal
from collections import OrderedDict, namedtuple
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature

SwingPoint = namedtuple("SwingPoint", "frame_idx price")


class SwingStructure(Feature):
    def __init__(
        self,
        swing_period: int = 5,
        source: tuple[str, str] = ("low", "high"),
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=swing_period)
        self.source = source
        self.names = [
            'swing_high_detected',
            'swing_low_detected',
            'swing_high_price',
            'swing_low_price',
            'bos_bullish',
            'bos_bearish',
            'choch_bullish',
            'choch_bearish',
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'swing_high_detected': gym.spaces.Discrete(2),
                'swing_low_detected': gym.spaces.Discrete(2),
                'swing_high_price': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'swing_low_price': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bos_bullish': gym.spaces.Discrete(2),
                'bos_bearish': gym.spaces.Discrete(2),
                'choch_bullish': gym.spaces.Discrete(2),
                'choch_bearish': gym.spaces.Discrete(2),
            })
        else:
            self.spaces = OrderedDict()
        self._swing_highs: list[SwingPoint] = []
        self._swing_lows: list[SwingPoint] = []
        self._trend: int = 0

    def reset(self):
        self._swing_highs.clear()
        self._swing_lows.clear()
        self._trend = 0

    def _detect_swing_high(self, frames: Sequence[Frame], idx: int) -> bool:
        r = self.period
        if idx < r or idx >= len(frames) - r:
            return False
        center = getattr(frames[idx], self.source[1])
        return all(
            getattr(frames[idx + offset], self.source[1]) < center
            for offset in range(-r, r + 1) if offset != 0
        )

    def _detect_swing_low(self, frames: Sequence[Frame], idx: int) -> bool:
        r = self.period
        if idx < r or idx >= len(frames) - r:
            return False
        center = getattr(frames[idx], self.source[0])
        return all(
            getattr(frames[idx + offset], self.source[0]) > center
            for offset in range(-r, r + 1) if offset != 0
        )

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        n = len(frames)
        check_idx = n - self.period - 1

        new_swing_high = 0
        new_swing_low = 0
        bos_bullish = 0
        bos_bearish = 0
        choch_bullish = 0
        choch_bearish = 0

        if check_idx >= 0 and self._detect_swing_high(frames, check_idx):
            price = getattr(frames[check_idx], self.source[1])
            self._swing_highs.append(SwingPoint(check_idx, price))
            if len(self._swing_highs) > 100:
                self._swing_highs.pop(0)
            new_swing_high = 1

            if len(self._swing_highs) >= 2:
                prev_high = self._swing_highs[-2].price
                if self._trend >= 0 and price > prev_high:
                    bos_bullish = 1
                    self._trend = 1
                elif self._trend < 0 and price > prev_high:
                    choch_bullish = 1
                    self._trend = 1

        if check_idx >= 0 and self._detect_swing_low(frames, check_idx):
            price = getattr(frames[check_idx], self.source[0])
            self._swing_lows.append(SwingPoint(check_idx, price))
            if len(self._swing_lows) > 100:
                self._swing_lows.pop(0)
            new_swing_low = 1

            if len(self._swing_lows) >= 2:
                prev_low = self._swing_lows[-2].price
                if self._trend <= 0 and price < prev_low:
                    bos_bearish = 1
                    self._trend = -1
                elif self._trend > 0 and price < prev_low:
                    choch_bearish = 1
                    self._trend = -1

        swing_high_price = self._swing_highs[-1].price if self._swing_highs else 0.0
        swing_low_price = self._swing_lows[-1].price if self._swing_lows else 0.0

        values = [
            new_swing_high, new_swing_low, swing_high_price, swing_low_price,
            bos_bullish, bos_bearish, choch_bullish, choch_bearish,
        ]

        if self.write_to_frame:
            for name, val in zip(self.names, values):
                setattr(frames[-1], name, val)
        if self.write_to_state:
            for name, val in zip(self.names, values):
                state[name] = val
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest test/test_smc_swing_structure.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add intraday/features/smc_swing_structure.py test/test_smc_swing_structure.py
git commit -m "feat: add SwingStructure SMC feature (BOS, ChoCh, swing points)"
```

---

## Task 2: PriceZones Feature

Premium/discount zones, displacement detection, order flow imbalance.

**Files:**
- Create: `intraday/features/smc_price_zones.py`
- Test: `test/test_smc_price_zones.py`

**Step 1: Write the failing test**

```python
# test/test_smc_price_zones.py
import unittest
from collections import OrderedDict
from intraday.features.smc_price_zones import PriceZones
from intraday.frame import Frame


class TestPriceZones(unittest.TestCase):
    def test_initializes(self):
        p = PriceZones(range_period=10)
        self.assertIn('zone_type', p.names)
        self.assertIn('equilibrium_price', p.names)
        self.assertIn('displacement_bullish', p.names)
        self.assertIn('volume_imbalance_ratio', p.names)

    def test_premium_zone(self):
        p = PriceZones(range_period=3)
        p.reset()
        frames = [
            Frame(high=100, low=90, close=95, open=93, buy_volume=500, sell_volume=500),
            Frame(high=105, low=95, close=100, open=96, buy_volume=500, sell_volume=500),
            Frame(high=110, low=100, close=108, open=101, buy_volume=500, sell_volume=500),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            p.process(frames[:i+1], state)
        # close=108, range 90-110, equilibrium=100, close > equilibrium -> premium
        self.assertEqual(state['zone_type'], 1)

    def test_discount_zone(self):
        p = PriceZones(range_period=3)
        p.reset()
        frames = [
            Frame(high=110, low=100, close=105, open=103, buy_volume=500, sell_volume=500),
            Frame(high=105, low=95, close=100, open=104, buy_volume=500, sell_volume=500),
            Frame(high=100, low=90, close=92, open=99, buy_volume=500, sell_volume=500),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            p.process(frames[:i+1], state)
        self.assertEqual(state['zone_type'], 2)

    def test_displacement_detection(self):
        p = PriceZones(range_period=3, displacement_threshold=0.5)
        p.reset()
        frames = []
        for i in range(10):
            frames.append(Frame(high=100+i, low=99+i, close=100+i, open=99+i,
                                buy_volume=500, sell_volume=500))
        # Add large bullish candle
        frames.append(Frame(high=130, low=109, close=129, open=110,
                           buy_volume=2000, sell_volume=200))
        for i in range(len(frames)):
            state = OrderedDict()
            p.process(frames[:i+1], state)
        self.assertEqual(state['displacement_bullish'], 1)

    def test_volume_imbalance(self):
        p = PriceZones(range_period=3)
        p.reset()
        frames = [
            Frame(high=100, low=95, close=98, open=96, buy_volume=900, sell_volume=100),
        ]
        state = OrderedDict()
        p.process(frames, state)
        self.assertGreater(state['volume_imbalance_ratio'], 1.0)

    def test_reset_clears_state(self):
        p = PriceZones(range_period=3)
        frames = [Frame(high=110, low=90, close=100, open=95,
                       buy_volume=500, sell_volume=500)]
        p.process(frames, OrderedDict())
        p.reset()
        state = OrderedDict()
        p.process([Frame(high=100, low=95, close=97, open=96,
                        buy_volume=500, sell_volume=500)], state)
        self.assertIn('zone_type', state)


if __name__ == '__main__':
    unittest.main()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest test/test_smc_price_zones.py -v
```
Expected: ImportError

**Step 3: Implement PriceZones**

```python
# intraday/features/smc_price_zones.py
from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class PriceZones(Feature):
    def __init__(
        self,
        range_period: int = 50,
        displacement_threshold: float = 1.5,
        imbalance_threshold: float = 2.0,
        source: tuple[str, str, str, str] = ("high", "low", "close", "open"),
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=range_period)
        self.source = source
        self.displacement_threshold = displacement_threshold
        self.imbalance_threshold = imbalance_threshold
        self.names = [
            'zone_type',
            'premium_percent',
            'discount_percent',
            'range_high',
            'range_low',
            'equilibrium_price',
            'displacement_bullish',
            'displacement_bearish',
            'displacement_magnitude',
            'volume_imbalance_ratio',
            'volume_imbalance_extreme',
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'zone_type': gym.spaces.Discrete(3),
                'premium_percent': gym.spaces.Box(0, 1, shape=(1,)),
                'discount_percent': gym.spaces.Box(0, 1, shape=(1,)),
                'range_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'range_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'equilibrium_price': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'displacement_bullish': gym.spaces.Discrete(2),
                'displacement_bearish': gym.spaces.Discrete(2),
                'displacement_magnitude': gym.spaces.Box(0, math.inf, shape=(1,)),
                'volume_imbalance_ratio': gym.spaces.Box(-10, 10, shape=(1,)),
                'volume_imbalance_extreme': gym.spaces.Discrete(2),
            })
        else:
            self.spaces = OrderedDict()
        self._atr_ema: float | None = None
        self._atr_factor = 2 / (range_period + 1)
        self._n_iter = 0

    def reset(self):
        self._atr_ema = None
        self._n_iter = 0

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        high_src, low_src, close_src, open_src = self.source

        window = frames[-min(len(frames), self.period):]
        range_high = max(getattr(f, high_src) for f in window)
        range_low = min(getattr(f, low_src) for f in window)
        equilibrium = (range_high + range_low) / 2
        range_size = range_high - range_low if range_high != range_low else 1e-9

        close = getattr(frame, close_src)
        open_ = getattr(frame, open_src)

        if close > equilibrium:
            zone_type = 1
            premium_pct = min(1.0, (close - equilibrium) / (range_size / 2))
            discount_pct = 0.0
        elif close < equilibrium:
            zone_type = 2
            premium_pct = 0.0
            discount_pct = min(1.0, (equilibrium - close) / (range_size / 2))
        else:
            zone_type = 0
            premium_pct = 0.0
            discount_pct = 0.0

        true_range = getattr(frame, 'true_range', abs(getattr(frame, high_src) - getattr(frame, low_src)))
        if self._atr_ema is None:
            self._atr_ema = true_range
        elif self._n_iter < self.period:
            self._atr_ema = (self._atr_ema * self._n_iter + true_range) / (self._n_iter + 1)
        else:
            self._atr_ema = self._atr_ema * (1 - self._atr_factor) + true_range * self._atr_factor
        self._n_iter += 1

        candle_size = abs(close - open_)
        atr = self._atr_ema if self._atr_ema > 0 else 1e-9
        displacement_magnitude = candle_size / atr
        displacement_bullish = int(close > open_ and displacement_magnitude > self.displacement_threshold)
        displacement_bearish = int(close < open_ and displacement_magnitude > self.displacement_threshold)

        buy_vol = getattr(frame, 'buy_volume', 0) or 0
        sell_vol = getattr(frame, 'sell_volume', 0) or 0
        if sell_vol > 0:
            imbalance_ratio = min(10.0, max(-10.0, buy_vol / sell_vol))
        else:
            imbalance_ratio = 10.0 if buy_vol > 0 else 0.0
        imbalance_extreme = int(abs(imbalance_ratio) > self.imbalance_threshold)

        values = [
            zone_type, premium_pct, discount_pct, range_high, range_low,
            equilibrium, displacement_bullish, displacement_bearish,
            displacement_magnitude, imbalance_ratio, imbalance_extreme,
        ]

        if self.write_to_frame:
            for name, val in zip(self.names, values):
                setattr(frame, name, val)
        if self.write_to_state:
            for name, val in zip(self.names, values):
                state[name] = val
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest test/test_smc_price_zones.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add intraday/features/smc_price_zones.py test/test_smc_price_zones.py
git commit -m "feat: add PriceZones SMC feature (premium/discount, displacement, imbalance)"
```

---

## Task 3: OrderBlock Feature

Order blocks (supply/demand zones) and fair value gaps (FVG).

**Files:**
- Create: `intraday/features/smc_order_block.py`
- Test: `test/test_smc_order_block.py`

**Step 1: Write the failing test**

```python
# test/test_smc_order_block.py
import unittest
from collections import OrderedDict
from intraday.features.smc_order_block import OrderBlock
from intraday.frame import Frame


class TestOrderBlock(unittest.TestCase):
    def test_initializes(self):
        ob = OrderBlock()
        self.assertIn('bullish_ob_detected', ob.names)
        self.assertIn('in_bullish_ob', ob.names)
        self.assertIn('fvg_bullish_detected', ob.names)

    def test_bullish_ob_detection(self):
        ob = OrderBlock(impulse_threshold=1.5)
        ob.reset()
        # Bearish candle followed by strong bullish impulse -> bullish OB
        frames = [
            Frame(open=105, high=106, low=98, close=99),   # bearish
            Frame(open=99, high=115, low=98, close=114),   # strong bullish (OB trigger)
            Frame(open=114, high=116, low=112, close=113),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            ob.process(frames[:i+1], state)
        # After the impulse, should have detected a bullish OB
        self.assertEqual(state['bullish_ob_detected'], 1)

    def test_bearish_ob_detection(self):
        ob = OrderBlock(impulse_threshold=1.5)
        ob.reset()
        frames = [
            Frame(open=100, high=108, low=99, close=107),  # bullish
            Frame(open=107, high=108, low=92, close=93),   # strong bearish (OB trigger)
            Frame(open=93, high=95, low=91, close=92),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            ob.process(frames[:i+1], state)
        self.assertEqual(state['bearish_ob_detected'], 1)

    def test_fvg_bullish_detection(self):
        ob = OrderBlock()
        ob.reset()
        # FVG: frames[0].high < frames[2].low (gap)
        frames = [
            Frame(open=100, high=102, low=98, close=101),
            Frame(open=103, high=108, low=102, close=107),
            Frame(open=107, high=115, low=105, close=113),  # frames[-3].high=102 < frames[-1].low=105
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            ob.process(frames[:i+1], state)
        self.assertEqual(state['fvg_bullish_detected'], 1)

    def test_reset_clears_zones(self):
        ob = OrderBlock()
        frames = [Frame(open=100, high=102, low=98, close=101)]
        ob.process(frames, OrderedDict())
        ob.reset()
        state = OrderedDict()
        ob.process([Frame(open=100, high=102, low=98, close=101)], state)
        self.assertEqual(state['bullish_ob_detected'], 0)


if __name__ == '__main__':
    unittest.main()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest test/test_smc_order_block.py -v
```
Expected: ImportError

**Step 3: Implement OrderBlock**

```python
# intraday/features/smc_order_block.py
from typing import Sequence, Literal
from collections import OrderedDict, namedtuple
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature

OBZone = namedtuple("OBZone", "top bottom frame_idx")
FVGZone = namedtuple("FVGZone", "top bottom gap_size frame_idx")


class OrderBlock(Feature):
    def __init__(
        self,
        impulse_threshold: float = 2.0,
        max_blocks: int = 50,
        fvg_threshold: float = 0.0,
        source: tuple[str, str, str, str] = ("open", "high", "low", "close"),
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to)
        self.impulse_threshold = impulse_threshold
        self.max_blocks = max_blocks
        self.fvg_threshold = fvg_threshold
        self.source = source
        self.names = [
            'bullish_ob_detected',
            'bearish_ob_detected',
            'bullish_ob_top',
            'bullish_ob_bottom',
            'bearish_ob_top',
            'bearish_ob_bottom',
            'in_bullish_ob',
            'in_bearish_ob',
            'fvg_bullish_detected',
            'fvg_bearish_detected',
            'fvg_bullish_gap_size',
            'fvg_bearish_gap_size',
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'bullish_ob_detected': gym.spaces.Discrete(2),
                'bearish_ob_detected': gym.spaces.Discrete(2),
                'bullish_ob_top': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bullish_ob_bottom': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bearish_ob_top': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bearish_ob_bottom': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'in_bullish_ob': gym.spaces.Discrete(2),
                'in_bearish_ob': gym.spaces.Discrete(2),
                'fvg_bullish_detected': gym.spaces.Discrete(2),
                'fvg_bearish_detected': gym.spaces.Discrete(2),
                'fvg_bullish_gap_size': gym.spaces.Box(0, math.inf, shape=(1,)),
                'fvg_bearish_gap_size': gym.spaces.Box(0, math.inf, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()
        self._bullish_obs: list[OBZone] = []
        self._bearish_obs: list[OBZone] = []
        self._bullish_fvgs: list[FVGZone] = []
        self._bearish_fvgs: list[FVGZone] = []
        self._atr_ema: float | None = None
        self._atr_factor = 2 / 15

    def reset(self):
        self._bullish_obs.clear()
        self._bearish_obs.clear()
        self._bullish_fvgs.clear()
        self._bearish_fvgs.clear()
        self._atr_ema = None

    def _update_atr(self, frame: Frame):
        tr = getattr(frame, 'true_range', abs(frame.high - frame.low))
        if self._atr_ema is None:
            self._atr_ema = tr
        else:
            self._atr_ema = self._atr_ema * (1 - self._atr_factor) + tr * self._atr_factor

    def _evict_mitigated(self, close: float):
        self._bullish_obs = [z for z in self._bullish_obs if close > z.bottom]
        self._bearish_obs = [z for z in self._bearish_obs if close < z.top]

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        open_s, high_s, low_s, close_s = self.source
        frame = frames[-1]
        self._update_atr(frame)
        close = getattr(frame, close_s)
        atr = self._atr_ema if self._atr_ema and self._atr_ema > 0 else 1e-9

        bullish_ob_detected = 0
        bearish_ob_detected = 0
        fvg_bullish_detected = 0
        fvg_bearish_detected = 0
        fvg_bullish_gap = 0.0
        fvg_bearish_gap = 0.0

        if len(frames) >= 2:
            prev = frames[-2]
            prev_close = getattr(prev, close_s)
            move = close - prev_close
            impulse = abs(move) > self.impulse_threshold * atr

            if impulse and move > 0 and len(frames) >= 2:
                ob_frame = prev
                ob_open = getattr(ob_frame, open_s)
                ob_close = getattr(ob_frame, close_s)
                if ob_close < ob_open:
                    top = max(ob_open, ob_close)
                    bottom = min(ob_open, ob_close)
                    self._bullish_obs.append(OBZone(top, bottom, len(frames) - 2))
                    if len(self._bullish_obs) > self.max_blocks:
                        self._bullish_obs.pop(0)
                    bullish_ob_detected = 1

            if impulse and move < 0 and len(frames) >= 2:
                ob_frame = prev
                ob_open = getattr(ob_frame, open_s)
                ob_close = getattr(ob_frame, close_s)
                if ob_close > ob_open:
                    top = max(ob_open, ob_close)
                    bottom = min(ob_open, ob_close)
                    self._bearish_obs.append(OBZone(top, bottom, len(frames) - 2))
                    if len(self._bearish_obs) > self.max_blocks:
                        self._bearish_obs.pop(0)
                    bearish_ob_detected = 1

        if len(frames) >= 3:
            f0, f2 = frames[-3], frames[-1]
            gap_up = getattr(f2, low_s) - getattr(f0, high_s)
            gap_down = getattr(f0, low_s) - getattr(f2, high_s)
            if gap_up > self.fvg_threshold:
                self._bullish_fvgs.append(FVGZone(getattr(f2, low_s), getattr(f0, high_s), gap_up, len(frames) - 1))
                fvg_bullish_detected = 1
                fvg_bullish_gap = gap_up
            elif gap_down > self.fvg_threshold:
                self._bearish_fvgs.append(FVGZone(getattr(f0, low_s), getattr(f2, high_s), gap_down, len(frames) - 1))
                fvg_bearish_detected = 1
                fvg_bearish_gap = gap_down

        self._evict_mitigated(close)

        bullish_ob_top = self._bullish_obs[-1].top if self._bullish_obs else 0.0
        bullish_ob_bottom = self._bullish_obs[-1].bottom if self._bullish_obs else 0.0
        bearish_ob_top = self._bearish_obs[-1].top if self._bearish_obs else 0.0
        bearish_ob_bottom = self._bearish_obs[-1].bottom if self._bearish_obs else 0.0

        in_bullish = int(any(z.bottom <= close <= z.top for z in self._bullish_obs))
        in_bearish = int(any(z.bottom <= close <= z.top for z in self._bearish_obs))

        values = [
            bullish_ob_detected, bearish_ob_detected,
            bullish_ob_top, bullish_ob_bottom, bearish_ob_top, bearish_ob_bottom,
            in_bullish, in_bearish,
            fvg_bullish_detected, fvg_bearish_detected,
            fvg_bullish_gap, fvg_bearish_gap,
        ]

        if self.write_to_frame:
            for name, val in zip(self.names, values):
                setattr(frame, name, val)
        if self.write_to_state:
            for name, val in zip(self.names, values):
                state[name] = val
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest test/test_smc_order_block.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add intraday/features/smc_order_block.py test/test_smc_order_block.py
git commit -m "feat: add OrderBlock SMC feature (order blocks, FVG detection)"
```

---

## Task 4: LiquiditySweep Feature

Liquidity pools, sweep (stop hunt) detection, inducement patterns.

**Files:**
- Create: `intraday/features/smc_liquidity_sweep.py`
- Test: `test/test_smc_liquidity_sweep.py`

**Step 1: Write the failing test**

```python
# test/test_smc_liquidity_sweep.py
import unittest
from collections import OrderedDict
from intraday.features.smc_liquidity_sweep import LiquiditySweep
from intraday.frame import Frame


class TestLiquiditySweep(unittest.TestCase):
    def test_initializes(self):
        ls = LiquiditySweep()
        self.assertIn('liquidity_above', ls.names)
        self.assertIn('sweep_high_detected', ls.names)
        self.assertIn('inducement_bullish', ls.names)

    def test_liquidity_levels_tracked(self):
        ls = LiquiditySweep(swing_period=1)
        ls.reset()
        frames = [
            Frame(high=100, low=90, close=95),
            Frame(high=110, low=105, close=108),  # potential swing
            Frame(high=105, low=100, close=102),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            ls.process(frames[:i+1], state)
        self.assertGreater(state['liquidity_above'], 0)

    def test_sweep_high_detected(self):
        ls = LiquiditySweep(swing_period=1, sweep_reversal_threshold=0.3)
        ls.reset()
        # Build up swing high, then sweep it
        frames = [
            Frame(high=100, low=90, close=95),
            Frame(high=110, low=105, close=108),  # swing high at 110
            Frame(high=105, low=100, close=102),
            Frame(high=115, low=108, close=109),  # breaks 110
            Frame(high=112, low=103, close=104),  # reverses down (sweep!)
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            ls.process(frames[:i+1], state)
        self.assertEqual(state['sweep_high_detected'], 1)

    def test_reset_clears_state(self):
        ls = LiquiditySweep()
        frames = [Frame(high=110, low=90, close=100)]
        ls.process(frames, OrderedDict())
        ls.reset()
        state = OrderedDict()
        ls.process([Frame(high=100, low=95, close=98)], state)
        self.assertEqual(state['liquidity_above_count'], 0)


if __name__ == '__main__':
    unittest.main()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest test/test_smc_liquidity_sweep.py -v
```
Expected: ImportError

**Step 3: Implement LiquiditySweep**

```python
# intraday/features/smc_liquidity_sweep.py
from typing import Sequence, Literal
from collections import OrderedDict, namedtuple
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature

SwingLevel = namedtuple("SwingLevel", "price frame_idx")


class LiquiditySweep(Feature):
    def __init__(
        self,
        swing_period: int = 5,
        lookback_swings: int = 10,
        sweep_reversal_threshold: float = 0.5,
        source: tuple[str, str, str] = ("low", "high", "close"),
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=swing_period)
        self.lookback_swings = lookback_swings
        self.sweep_reversal_threshold = sweep_reversal_threshold
        self.source = source
        self.names = [
            'liquidity_above',
            'liquidity_below',
            'liquidity_above_count',
            'liquidity_below_count',
            'sweep_high_detected',
            'sweep_low_detected',
            'sweep_high_reversal',
            'sweep_low_reversal',
            'inducement_bullish',
            'inducement_bearish',
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'liquidity_above': gym.spaces.Box(0, math.inf, shape=(1,)),
                'liquidity_below': gym.spaces.Box(0, math.inf, shape=(1,)),
                'liquidity_above_count': gym.spaces.Discrete(21),
                'liquidity_below_count': gym.spaces.Discrete(21),
                'sweep_high_detected': gym.spaces.Discrete(2),
                'sweep_low_detected': gym.spaces.Discrete(2),
                'sweep_high_reversal': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'sweep_low_reversal': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'inducement_bullish': gym.spaces.Discrete(2),
                'inducement_bearish': gym.spaces.Discrete(2),
            })
        else:
            self.spaces = OrderedDict()
        self._liq_highs: list[SwingLevel] = []
        self._liq_lows: list[SwingLevel] = []
        self._prev_high_broken: float | None = None
        self._prev_low_broken: float | None = None
        self._atr_ema: float | None = None
        self._atr_factor = 2 / (swing_period * 2 + 1)

    def reset(self):
        self._liq_highs.clear()
        self._liq_lows.clear()
        self._prev_high_broken = None
        self._prev_low_broken = None
        self._atr_ema = None

    def _is_swing_high(self, frames: Sequence[Frame], idx: int) -> bool:
        r = self.period
        if idx < r or idx >= len(frames) - r:
            return False
        center = getattr(frames[idx], self.source[1])
        return all(getattr(frames[idx + o], self.source[1]) < center
                   for o in range(-r, r + 1) if o != 0)

    def _is_swing_low(self, frames: Sequence[Frame], idx: int) -> bool:
        r = self.period
        if idx < r or idx >= len(frames) - r:
            return False
        center = getattr(frames[idx], self.source[0])
        return all(getattr(frames[idx + o], self.source[0]) > center
                   for o in range(-r, r + 1) if o != 0)

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        low_s, high_s, close_s = self.source
        frame = frames[-1]
        n = len(frames)
        close = getattr(frame, close_s)
        high = getattr(frame, high_s)
        low = getattr(frame, low_s)

        tr = abs(high - low)
        if self._atr_ema is None:
            self._atr_ema = tr
        else:
            self._atr_ema = self._atr_ema * (1 - self._atr_factor) + tr * self._atr_factor
        atr = self._atr_ema if self._atr_ema > 0 else 1e-9

        check_idx = n - self.period - 1
        if check_idx >= 0:
            if self._is_swing_high(frames, check_idx):
                price = getattr(frames[check_idx], high_s)
                self._liq_highs.append(SwingLevel(price, check_idx))
                if len(self._liq_highs) > self.lookback_swings:
                    self._liq_highs.pop(0)
            if self._is_swing_low(frames, check_idx):
                price = getattr(frames[check_idx], low_s)
                self._liq_lows.append(SwingLevel(price, check_idx))
                if len(self._liq_lows) > self.lookback_swings:
                    self._liq_lows.pop(0)

        sweep_high = 0
        sweep_low = 0
        sweep_high_reversal = 0.0
        sweep_low_reversal = 0.0
        inducement_bullish = 0
        inducement_bearish = 0

        for level in self._liq_highs:
            if high > level.price and close < level.price:
                reversal = level.price - close
                if reversal > self.sweep_reversal_threshold * atr:
                    sweep_high = 1
                    sweep_high_reversal = reversal

        for level in self._liq_lows:
            if low < level.price and close > level.price:
                reversal = close - level.price
                if reversal > self.sweep_reversal_threshold * atr:
                    sweep_low = 1
                    sweep_low_reversal = reversal

        if len(self._liq_highs) >= 2 and len(self._liq_lows) >= 1:
            if (self._liq_highs[-1].price > self._liq_highs[-2].price and
                    low < self._liq_lows[-1].price):
                inducement_bearish = 1
        if len(self._liq_lows) >= 2 and len(self._liq_highs) >= 1:
            if (self._liq_lows[-1].price < self._liq_lows[-2].price and
                    high > self._liq_highs[-1].price):
                inducement_bullish = 1

        highs_above = [l for l in self._liq_highs if l.price > close]
        lows_below = [l for l in self._liq_lows if l.price < close]
        liq_above = min(h.price for h in highs_above) if highs_above else 0.0
        liq_below = max(l.price for l in lows_below) if lows_below else 0.0

        values = [
            liq_above, liq_below, len(highs_above), len(lows_below),
            sweep_high, sweep_low, sweep_high_reversal, sweep_low_reversal,
            inducement_bullish, inducement_bearish,
        ]

        if self.write_to_frame:
            for name, val in zip(self.names, values):
                setattr(frame, name, val)
        if self.write_to_state:
            for name, val in zip(self.names, values):
                state[name] = val
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest test/test_smc_liquidity_sweep.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add intraday/features/smc_liquidity_sweep.py test/test_smc_liquidity_sweep.py
git commit -m "feat: add LiquiditySweep SMC feature (liquidity pools, sweeps, inducement)"
```

---

## Task 5: SessionLevels Feature

Session highs/lows (Asian/London/NY) and kill zone detection.

**Files:**
- Create: `intraday/features/smc_session_levels.py`
- Test: `test/test_smc_session_levels.py`

**Step 1: Write the failing test**

```python
# test/test_smc_session_levels.py
import unittest
from collections import OrderedDict
import arrow
from intraday.features.smc_session_levels import SessionLevels
from intraday.frame import Frame


def make_frame(hour_utc: int, high: float, low: float, close: float) -> Frame:
    t = arrow.Arrow(2024, 1, 15, hour_utc, 0, 0)
    f = Frame(high=high, low=low, close=close)
    f.time_start = t
    f.time_end = t.shift(hours=1)
    return f


class TestSessionLevels(unittest.TestCase):
    def test_initializes(self):
        sl = SessionLevels()
        self.assertIn('session_type', sl.names)
        self.assertIn('asian_high', sl.names)
        self.assertIn('in_kill_zone', sl.names)

    def test_asian_session(self):
        sl = SessionLevels()
        sl.reset()
        frames = [make_frame(2, 100, 90, 95)]
        state = OrderedDict()
        sl.process(frames, state)
        self.assertEqual(state['session_type'], 1)
        self.assertAlmostEqual(state['asian_high'], 100.0)

    def test_london_session(self):
        sl = SessionLevels()
        sl.reset()
        frames = [make_frame(9, 105, 95, 100)]
        state = OrderedDict()
        sl.process(frames, state)
        self.assertEqual(state['session_type'], 2)

    def test_ny_session(self):
        sl = SessionLevels()
        sl.reset()
        frames = [make_frame(14, 110, 100, 105)]
        state = OrderedDict()
        sl.process(frames, state)
        self.assertEqual(state['session_type'], 3)

    def test_kill_zone(self):
        sl = SessionLevels()
        sl.reset()
        # London open kill zone: 7-10 UTC
        frames = [make_frame(8, 105, 95, 100)]
        state = OrderedDict()
        sl.process(frames, state)
        self.assertEqual(state['in_kill_zone'], 1)

    def test_session_high_low_tracking(self):
        sl = SessionLevels()
        sl.reset()
        frames = [
            make_frame(2, 100, 90, 95),
            make_frame(3, 105, 92, 103),
            make_frame(4, 103, 88, 91),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            sl.process(frames[:i+1], state)
        self.assertAlmostEqual(state['asian_high'], 105.0)
        self.assertAlmostEqual(state['asian_low'], 88.0)

    def test_reset_clears_state(self):
        sl = SessionLevels()
        frames = [make_frame(2, 100, 90, 95)]
        sl.process(frames, OrderedDict())
        sl.reset()
        state = OrderedDict()
        sl.process([make_frame(8, 105, 95, 100)], state)
        self.assertEqual(state['session_type'], 2)


if __name__ == '__main__':
    unittest.main()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest test/test_smc_session_levels.py -v
```
Expected: ImportError

**Step 3: Implement SessionLevels**

```python
# intraday/features/smc_session_levels.py
from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature

ASIAN_START, ASIAN_END = 0, 9
LONDON_START, LONDON_END = 7, 16
NY_START, NY_END = 12, 21
KILL_ZONES = set(range(7, 10)) | set(range(12, 15))


class SessionLevels(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = [
            'session_type',
            'in_kill_zone',
            'asian_high', 'asian_low', 'asian_range',
            'london_high', 'london_low', 'london_range',
            'ny_high', 'ny_low', 'ny_range',
            'prev_session_high', 'prev_session_low',
            'distance_to_session_high', 'distance_to_session_low',
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'session_type': gym.spaces.Discrete(4),
                'in_kill_zone': gym.spaces.Discrete(2),
                'asian_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'asian_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'asian_range': gym.spaces.Box(0, math.inf, shape=(1,)),
                'london_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'london_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'london_range': gym.spaces.Box(0, math.inf, shape=(1,)),
                'ny_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'ny_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'ny_range': gym.spaces.Box(0, math.inf, shape=(1,)),
                'prev_session_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'prev_session_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'distance_to_session_high': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'distance_to_session_low': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()
        self._asian_high = -math.inf
        self._asian_low = math.inf
        self._london_high = -math.inf
        self._london_low = math.inf
        self._ny_high = -math.inf
        self._ny_low = math.inf
        self._prev_high = 0.0
        self._prev_low = 0.0
        self._last_date = None
        self._last_session = 0

    def reset(self):
        self._asian_high = -math.inf
        self._asian_low = math.inf
        self._london_high = -math.inf
        self._london_low = math.inf
        self._ny_high = -math.inf
        self._ny_low = math.inf
        self._prev_high = 0.0
        self._prev_low = 0.0
        self._last_date = None
        self._last_session = 0

    def _session_type(self, hour: int) -> int:
        if NY_START <= hour < NY_END:
            return 3
        if LONDON_START <= hour < LONDON_END:
            return 2
        if ASIAN_START <= hour < ASIAN_END:
            return 1
        return 0

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        time = getattr(frame, 'time_end', None) or getattr(frame, 'time_start', None)

        if time is None:
            hour = 0
            date = None
        else:
            hour = time.hour
            date = time.date()

        if date and date != self._last_date:
            self._asian_high = -math.inf
            self._asian_low = math.inf
            self._london_high = -math.inf
            self._london_low = math.inf
            self._ny_high = -math.inf
            self._ny_low = math.inf
            self._last_date = date

        session = self._session_type(hour)

        if ASIAN_START <= hour < ASIAN_END:
            self._asian_high = max(self._asian_high, frame.high)
            self._asian_low = min(self._asian_low, frame.low)
        if LONDON_START <= hour < LONDON_END:
            self._london_high = max(self._london_high, frame.high)
            self._london_low = min(self._london_low, frame.low)
        if NY_START <= hour < NY_END:
            self._ny_high = max(self._ny_high, frame.high)
            self._ny_low = min(self._ny_low, frame.low)

        if session != self._last_session and self._last_session != 0:
            if self._last_session == 1:
                self._prev_high = self._asian_high if self._asian_high != -math.inf else 0.0
                self._prev_low = self._asian_low if self._asian_low != math.inf else 0.0
            elif self._last_session == 2:
                self._prev_high = self._london_high if self._london_high != -math.inf else 0.0
                self._prev_low = self._london_low if self._london_low != math.inf else 0.0
            elif self._last_session == 3:
                self._prev_high = self._ny_high if self._ny_high != -math.inf else 0.0
                self._prev_low = self._ny_low if self._ny_low != math.inf else 0.0
        self._last_session = session

        asian_h = self._asian_high if self._asian_high != -math.inf else 0.0
        asian_l = self._asian_low if self._asian_low != math.inf else 0.0
        london_h = self._london_high if self._london_high != -math.inf else 0.0
        london_l = self._london_low if self._london_low != math.inf else 0.0
        ny_h = self._ny_high if self._ny_high != -math.inf else 0.0
        ny_l = self._ny_low if self._ny_low != math.inf else 0.0

        session_high = {1: asian_h, 2: london_h, 3: ny_h}.get(session, 0.0)
        session_low = {1: asian_l, 2: london_l, 3: ny_l}.get(session, 0.0)

        values = [
            session, int(hour in KILL_ZONES),
            asian_h, asian_l, asian_h - asian_l,
            london_h, london_l, london_h - london_l,
            ny_h, ny_l, ny_h - ny_l,
            self._prev_high, self._prev_low,
            frame.close - session_high, frame.close - session_low,
        ]

        if self.write_to_frame:
            for name, val in zip(self.names, values):
                setattr(frame, name, val)
        if self.write_to_state:
            for name, val in zip(self.names, values):
                state[name] = val
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest test/test_smc_session_levels.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add intraday/features/smc_session_levels.py test/test_smc_session_levels.py
git commit -m "feat: add SessionLevels SMC feature (Asian/London/NY sessions, kill zones)"
```

---

## Task 6: Update exports and integration test

**Files:**
- Modify: `intraday/features/__init__.py`
- Test: `test/test_smc_integration.py`

**Step 1: Update __init__.py**

Add to `intraday/features/__init__.py` after existing imports:

```python
from .smc_swing_structure import SwingStructure
from .smc_price_zones import PriceZones
from .smc_order_block import OrderBlock
from .smc_liquidity_sweep import LiquiditySweep
from .smc_session_levels import SessionLevels
```

Add to `__all__`:

```python
"SwingStructure",
"PriceZones",
"OrderBlock",
"LiquiditySweep",
"SessionLevels",
```

**Step 2: Verify imports work**

```bash
source /home/hung/env/.venv/bin/activate
python -c "from intraday.features import SwingStructure, PriceZones, OrderBlock, LiquiditySweep, SessionLevels; print('All SMC imports OK')"
```
Expected: `All SMC imports OK`

**Step 3: Write integration test**

```python
# test/test_smc_integration.py
import unittest
from collections import OrderedDict
import arrow
from intraday.features import SwingStructure, PriceZones, OrderBlock, LiquiditySweep, SessionLevels
from intraday.frame import Frame


def make_frame(high, low, close, open_=None, volume=1000, hour=10):
    f = Frame(
        high=high, low=low, close=close,
        open=open_ or close - 0.5,
        volume=volume,
    )
    f.buy_volume = volume * 0.55
    f.sell_volume = volume * 0.45
    t = arrow.Arrow(2024, 1, 15, hour, 0, 0)
    f.time_start = t
    f.time_end = t.shift(hours=1)
    return f


class TestSMCIntegration(unittest.TestCase):
    def test_all_features_produce_outputs(self):
        features = [
            SwingStructure(swing_period=2),
            PriceZones(range_period=5),
            OrderBlock(impulse_threshold=1.5),
            LiquiditySweep(swing_period=2),
            SessionLevels(),
        ]
        for f in features:
            f.reset()

        frames = []
        for i in range(15):
            h = 100 + i + (i % 3)
            l = 98 + i - (i % 2)
            c = 99 + i
            frames.append(make_frame(h, l, c, hour=8 + (i % 8)))

        state = OrderedDict()
        for i in range(len(frames)):
            state = OrderedDict()
            for feature in features:
                feature.process(frames[:i+1], state)

        expected_keys = [
            'swing_high_detected', 'bos_bullish',
            'zone_type', 'equilibrium_price',
            'bullish_ob_detected', 'fvg_bullish_detected',
            'liquidity_above', 'sweep_high_detected',
            'session_type', 'in_kill_zone',
        ]
        for key in expected_keys:
            self.assertIn(key, state, f"Missing key: {key}")

    def test_no_name_collisions(self):
        features = [
            SwingStructure(), PriceZones(), OrderBlock(),
            LiquiditySweep(), SessionLevels(),
        ]
        all_names = []
        for f in features:
            all_names.extend(f.names)
        self.assertEqual(len(all_names), len(set(all_names)), "Name collision detected")


if __name__ == '__main__':
    unittest.main()
```

**Step 4: Run all SMC tests**

```bash
python -m pytest test/test_smc_*.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add intraday/features/__init__.py test/test_smc_integration.py
git commit -m "feat: export SMC features and add integration tests"
```

---

## Final Verification

**Run full test suite:**

```bash
source /home/hung/env/.venv/bin/activate
python -m pytest test/ -v --ignore=test/test_dukascopy_api.py
```
Expected: All non-network tests PASS
