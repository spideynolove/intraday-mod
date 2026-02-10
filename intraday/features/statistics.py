from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class BETA(Feature):
    def __init__(
        self,
        period: int = 5,
        source1: str = 'close',
        source2: str = 'volume',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source1 = source1
        self.source2 = source2
        self.names = [f'beta_{period}_{source1}_{source2}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        if window_size < 2:
            beta = 0.0
        else:
            window = frames[-window_size:]
            x_values = [getattr(f, self.source1) for f in window]
            y_values = [getattr(f, self.source2) for f in window]

            x_mean = sum(x_values) / len(x_values)
            y_mean = sum(y_values) / len(y_values)

            covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            variance_x = sum((x - x_mean) ** 2 for x in x_values)

            if variance_x == 0:
                beta = 0.0
            else:
                beta = covariance / variance_x

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], beta)
        if self.write_to_state:
            state[self.names[0]] = beta


class CORREL(Feature):
    def __init__(
        self,
        period: int = 30,
        source1: str = 'close',
        source2: str = 'volume',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source1 = source1
        self.source2 = source2
        self.names = [f'correl_{period}_{source1}_{source2}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-1, 1, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        if window_size < 2:
            correl = 0.0
        else:
            window = frames[-window_size:]
            x_values = [getattr(f, self.source1) for f in window]
            y_values = [getattr(f, self.source2) for f in window]

            x_mean = sum(x_values) / len(x_values)
            y_mean = sum(y_values) / len(y_values)

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            x_variance = sum((x - x_mean) ** 2 for x in x_values)
            y_variance = sum((y - y_mean) ** 2 for y in y_values)

            denominator = math.sqrt(x_variance * y_variance)
            if denominator == 0:
                correl = 0.0
            else:
                correl = numerator / denominator

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], correl)
        if self.write_to_state:
            state[self.names[0]] = correl
