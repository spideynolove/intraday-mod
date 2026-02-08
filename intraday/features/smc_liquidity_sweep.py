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
        self._atr_ema: float | None = None
        self._atr_factor = 2 / (swing_period * 2 + 1)

    def reset(self):
        self._liq_highs.clear()
        self._liq_lows.clear()
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
