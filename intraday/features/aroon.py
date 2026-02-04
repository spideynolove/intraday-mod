from typing import Sequence, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class Aroon(Feature):
    def __init__(
        self,
        period: int = 25,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.names = [f'aroon_up_{period}', f'aroon_down_{period}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, 100, shape=(1,)),
                self.names[1]: gym.spaces.Box(0, 100, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period + 1)
        window = frames[-window_size:]

        highs = [f.high for f in window]
        lows = [f.low for f in window]

        periods_since_high = len(highs) - 1 - highs.index(max(highs))
        periods_since_low = len(lows) - 1 - lows.index(min(lows))

        aroon_up = 100 * (self.period - periods_since_high) / self.period
        aroon_down = 100 * (self.period - periods_since_low) / self.period

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], aroon_up)
            setattr(frames[-1], self.names[1], aroon_down)
        if self.write_to_state:
            state[self.names[0]] = aroon_up
            state[self.names[1]] = aroon_down
