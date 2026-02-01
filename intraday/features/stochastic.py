from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class Stochastic(Feature):
    def __init__(
        self,
        period: int = 10,
        source: Tuple[str, str, str] = ("low", "high", "close"),
        write_to: Literal["frame", "state", "both"] = "both",
    ):
        assert isinstance(period, int) and period > 0
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, Tuple) and len(source) == 3
        self.source = source
        name = f"stoch_{self.period}_{self.source[0]}_{self.source[1]}_{self.source[2]}"
        self.names = [name]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {
                    name: gym.spaces.Box(-0.5, 0.5, shape=(1,))
                    for name in self.names
                }
            )
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        result = self.calculate(frames[-self.period :], self.source)
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result

    @staticmethod
    def calculate(frames: Sequence[Frame], source: Tuple[str, str, str]):
        highest, lowest = None, None
        for frame in frames:
            low = getattr(frame, source[0])
            high = getattr(frame, source[1])
            if highest is None or highest < high:
                highest = high
            if lowest is None or lowest > low:
                lowest = low
        delta = highest - lowest
        close = getattr(frames[-1], source[2])
        return (close - lowest) / delta - 0.5 if delta != 0 else 0.0
