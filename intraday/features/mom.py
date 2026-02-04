from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class MOM(Feature):
    def __init__(
        self,
        period: int = 10,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.names = [f'mom_{period}_{source}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if len(frames) <= self.period:
            momentum = 0.0
        else:
            current = getattr(frames[-1], self.source)
            past = getattr(frames[-self.period - 1], self.source)
            momentum = current - past

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], momentum)
        if self.write_to_state:
            state[self.names[0]] = momentum
