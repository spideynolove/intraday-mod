from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class HeikenAshi(Feature):
    def __init__(
        self,
        source: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
        write_to: Literal["frame", "state", "both"] = "both",
    ):
        super().__init__(write_to=write_to)
        assert isinstance(source, Tuple) and len(source) == 4
        self.source = source
        self.names = [
            f"heiken_{self.source[0]}",
            f"heiken_{self.source[1]}",
            f"heiken_{self.source[2]}",
            f"heiken_{self.source[3]}",
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {
                    name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                    for name in self.names
                }
            )
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        last_open = getattr(last_frame, self.source[0])
        last_high = getattr(last_frame, self.source[1])
        last_low = getattr(last_frame, self.source[2])
        last_close = getattr(last_frame, self.source[3])
        if len(frames) > 1:
            prev_frame = frames[-2]
            prev_open = getattr(prev_frame, self.source[0])
            prev_close = getattr(prev_frame, self.source[3])
            open = (prev_open + prev_close) / 2
        else:
            open = last_open
        high = last_high
        low = last_low
        close = (last_open + last_high + last_low + last_close) / 4
        if self.write_to_frame:
            setattr(last_frame, self.names[0], open)
            setattr(last_frame, self.names[1], high)
            setattr(last_frame, self.names[2], low)
            setattr(last_frame, self.names[3], close)
        if self.write_to_state:
            state[self.names[0]] = open
            state[self.names[1]] = high
            state[self.names[2]] = low
            state[self.names[3]] = close

    def __repr__(self):
        return f"{self.__class__.__name__}(source={self.source}, write_to={self.write_to})"
