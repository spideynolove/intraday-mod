from typing import Sequence, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class WILLR(Feature):
    def __init__(
        self,
        period: int = 14,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.names = [f'willr_{period}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-100, 0, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        pass

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        window = frames[-window_size:]

        highest = max(f.high for f in window)
        lowest = min(f.low for f in window)
        close = frames[-1].close

        if highest == lowest:
            willr = -50.0
        else:
            willr = -100 * (highest - close) / (highest - lowest)

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], willr)
        if self.write_to_state:
            state[self.names[0]] = willr

    def __repr__(self):
        return f"WILLR(period={self.period}, write_to={self.write_to})"
