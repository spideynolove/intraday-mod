from typing import Sequence, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class EfficiencyRatio(Feature):
    def __init__(
        self,
        period: int = 10,
        source: str = "close",
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        assert isinstance(period, int) and period > 0
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, str)
        self.source = source
        name = f"efficiency_ratio_{self.period}_{self.source}"
        self.names = [name]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {name: gym.spaces.Box(0, 1, shape=(1,)) for name in self.names}
            )
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        result = EfficiencyRatio.calculate(frames[-self.period :], self.source)
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result

    @staticmethod
    def calculate(frames: Sequence[Frame], source: str):
        prev_value = getattr(frames[0], source)
        value = getattr(frames[-1], source)
        change = value - prev_value
        volatility = 0.0
        for frame in frames[1:]:
            value = getattr(frame, source)
            volatility += abs(value - prev_value)
            prev_value = value
        return change / volatility if volatility != 0 else 0.0

    def __repr__(self):
        return f"{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})"
