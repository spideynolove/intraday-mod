from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class MACD(Feature):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.source = source
        self.names = ['macd', 'macd_signal', 'macd_histogram']

        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None
        self._fast_factor = 2 / (fast_period + 1)
        self._slow_factor = 2 / (slow_period + 1)
        self._signal_factor = 2 / (signal_period + 1)
        self._n_iter = 0

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'macd': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'macd_signal': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'macd_histogram': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None
        self._fast_factor = 2 / (self.fast_period + 1)
        self._slow_factor = 2 / (self.slow_period + 1)
        self._signal_factor = 2 / (self.signal_period + 1)
        self._n_iter = 0

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        value = getattr(frame, self.source)

        if self._fast_ema is None:
            self._fast_ema = value
        elif self._n_iter < self.fast_period:
            self._fast_ema = (self._fast_ema * self._n_iter + value) / (self._n_iter + 1)
        else:
            self._fast_ema = self._fast_ema * (1 - self._fast_factor) + value * self._fast_factor

        if self._slow_ema is None:
            self._slow_ema = value
        elif self._n_iter < self.slow_period:
            self._slow_ema = (self._slow_ema * self._n_iter + value) / (self._n_iter + 1)
        else:
            self._slow_ema = self._slow_ema * (1 - self._slow_factor) + value * self._slow_factor

        macd_value = self._fast_ema - self._slow_ema

        if self._signal_ema is None:
            self._signal_ema = macd_value
        elif self._n_iter < self.signal_period:
            self._signal_ema = (self._signal_ema * self._n_iter + macd_value) / (self._n_iter + 1)
        else:
            self._signal_ema = self._signal_ema * (1 - self._signal_factor) + macd_value * self._signal_factor

        histogram = macd_value - self._signal_ema

        self._n_iter += 1

        if self.write_to_frame:
            setattr(frame, 'macd', macd_value)
            setattr(frame, 'macd_signal', self._signal_ema)
            setattr(frame, 'macd_histogram', histogram)
        if self.write_to_state:
            state['macd'] = macd_value
            state['macd_signal'] = self._signal_ema
            state['macd_histogram'] = histogram

    def __repr__(self):
        return f"MACD(fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}, write_to={self.write_to})"
