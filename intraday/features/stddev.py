from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class STDDEV(Feature):
    def __init__(
        self,
        period: int = 5,
        source: str = 'close',
        nbdev: float = 1.0,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.nbdev = nbdev
        self.names = [f'stddev_{period}_{source}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        values = [getattr(f, self.source) for f in frames[-window_size:]]

        if len(values) > 1:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            stddev = math.sqrt(variance) * self.nbdev
        else:
            stddev = 0.0

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], stddev)
        if self.write_to_state:
            state[self.names[0]] = stddev


class VAR(Feature):
    def __init__(
        self,
        period: int = 5,
        source: str = 'close',
        nbdev: float = 1.0,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.nbdev = nbdev
        self.names = [f'var_{period}_{source}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        values = [getattr(f, self.source) for f in frames[-window_size:]]

        if len(values) > 1:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            variance = variance * (self.nbdev ** 2)
        else:
            variance = 0.0

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], variance)
        if self.write_to_state:
            state[self.names[0]] = variance
