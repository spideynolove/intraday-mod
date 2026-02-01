from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class MarketDimension(Feature):
    def __init__(
        self,
        period: int = 10,
        source: Tuple[str, str] = ("low", "high"),
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        assert isinstance(period, int) and period > 0
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, Tuple) and len(source) == 2
        self.source = source
        name = (
            f"market_dimension_{self.period}_{self.source[0]}_{self.source[1]}"
        )
        self.names = [name]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {name: gym.spaces.Box(0, 1, shape=(1,)) for name in self.names}
            )
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        result = MarketDimension.calculate(frames[-self.period :], self.source)
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result

    @staticmethod
    def calculate(frames: Sequence[Frame], source: Tuple[str, str]):
        period = len(frames)
        highest, lowest, SN = None, None, 0.0
        for frame in frames:
            low = getattr(frame, source[0])
            high = getattr(frame, source[1])
            if highest is None or highest < high:
                highest = high
            if lowest is None or lowest > low:
                lowest = low
            SN += high - low
        S1 = highest - lowest
        S2 = S1 * period
        return (SN - S1) / (S2 - S1) if S1 > 0 and period > 1 else 0.5

    def __repr__(self):
        return f"{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})"
