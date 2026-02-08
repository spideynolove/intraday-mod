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
