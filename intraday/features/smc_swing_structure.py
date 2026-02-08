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
