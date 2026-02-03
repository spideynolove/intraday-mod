from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class BollingerBands(Feature):
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.std_dev = std_dev
        self.source = source
        self.names = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width']

        self._sma = None
        self._n_iter = 0
        self._ema_factor = 2 / (period + 1)

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'bb_middle': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bb_upper': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bb_lower': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bb_width': gym.spaces.Box(0, math.inf, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._sma = None
        self._n_iter = 0
        self._ema_factor = 2 / (self.period + 1)

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        value = getattr(frame, self.source)

        if self._sma is None:
            self._sma = value
        elif self._n_iter < self.period:
            self._sma = (self._sma * self._n_iter + value) / (self._n_iter + 1)
        else:
            self._sma = self._sma * (1 - self._ema_factor) + value * self._ema_factor

        window_size = min(len(frames), self.period)
        values = [getattr(f, self.source) for f in frames[-window_size:]]

        if len(values) > 1:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = math.sqrt(variance)
        else:
            std = 0.0

        middle = self._sma
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)
        width = upper - lower

        self._n_iter += 1

        if self.write_to_frame:
            setattr(frame, 'bb_middle', middle)
            setattr(frame, 'bb_upper', upper)
            setattr(frame, 'bb_lower', lower)
            setattr(frame, 'bb_width', width)
        if self.write_to_state:
            state['bb_middle'] = middle
            state['bb_upper'] = upper
            state['bb_lower'] = lower
            state['bb_width'] = width

    def __repr__(self):
        return f"BollingerBands(period={self.period}, std_dev={self.std_dev}, source={self.source}, write_to={self.write_to})"
