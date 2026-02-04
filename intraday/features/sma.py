from typing import Sequence, Union, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class SMA(Feature):
    def __init__(
        self,
        period: int,
        source: Union[str, Sequence[str]],
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = [source] if isinstance(source, str) else list(source)
        self.names = [f'sma_{period}_{name}' for name in self.source]

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                for name in self.names
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        window = frames[-window_size:]

        for i, src in enumerate(self.source):
            values = [getattr(f, src) for f in window]
            sma = sum(values) / len(values)

            if self.write_to_frame:
                setattr(frames[-1], self.names[i], sma)
            if self.write_to_state:
                state[self.names[i]] = sma


class WMA(Feature):
    def __init__(
        self,
        period: int,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.names = [f'wma_{period}_{source}']
        self._weights_sum = period * (period + 1) / 2

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        window = frames[-window_size:]
        values = [getattr(f, self.source) for f in window]

        weighted_sum = sum(v * (i + 1) for i, v in enumerate(values))
        actual_weights_sum = window_size * (window_size + 1) / 2
        wma = weighted_sum / actual_weights_sum

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], wma)
        if self.write_to_state:
            state[self.names[0]] = wma


class TRIMA(Feature):
    def __init__(
        self,
        period: int,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.names = [f'trima_{period}_{source}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        window = frames[-window_size:]
        values = [getattr(f, self.source) for f in window]

        n = len(values)
        if n % 2 == 0:
            mid = n // 2
            weights = list(range(1, mid + 1)) + list(range(mid, 0, -1))
        else:
            mid = (n + 1) // 2
            weights = list(range(1, mid + 1)) + list(range(mid - 1, 0, -1))

        weighted_sum = sum(v * w for v, w in zip(values, weights))
        trima = weighted_sum / sum(weights)

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], trima)
        if self.write_to_state:
            state[self.names[0]] = trima
