from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class VI(Feature):
    def __init__(
        self,
        source: Tuple[str, str] = ("close", "volume"),
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=2)
        assert isinstance(source, Tuple) and len(source) == 2
        self.source = source
        self.names = [
            f"pvi_{self.source[0]}_{self.source[1]}",
            f"nvi_{self.source[0]}_{self.source[1]}",
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
        self.pvi = 100.0
        self.nvi = 100.0

    def reset(self):
        self.pvi = 100.0
        self.nvi = 100.0

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        close, volume = getattr(frame, self.source[0]), getattr(
            frame, self.source[1]
        )
        prev_close = (
            getattr(frames[-2], self.source[0]) if len(frames) > 1 else close
        )
        prev_volume = (
            getattr(frames[-2], self.source[1]) if len(frames) > 1 else volume
        )
        change = (close - prev_close) / prev_close
        if volume > prev_volume:
            self.pvi *= 1 + change
        elif volume < prev_volume:
            self.nvi *= 1 + change
        if self.write_to_frame:
            setattr(frame, self.names[0], self.pvi)
            setattr(frame, self.names[1], self.nvi)
        if self.write_to_state:
            state[self.names[0]] = self.pvi
            state[self.names[1]] = self.nvi
        return self.pvi, self.nvi
