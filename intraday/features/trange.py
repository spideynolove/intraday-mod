from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class TRANGE(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = ['trange']
        self._prev_close = None

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'trange': gym.spaces.Box(0, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._prev_close = None

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]

        if self._prev_close is None:
            trange = frame.high - frame.low
        else:
            trange = max(
                frame.high - frame.low,
                abs(frame.high - self._prev_close),
                abs(frame.low - self._prev_close)
            )

        self._prev_close = frame.close

        if self.write_to_frame:
            setattr(frame, 'trange', trange)
        if self.write_to_state:
            state['trange'] = trange
