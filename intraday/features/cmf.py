from typing import Sequence, Tuple, Literal
from collections import OrderedDict
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class CMF(Feature):
    def __init__(
        self,
        period: int = 20,
        source: Tuple[str, str, str, str] = ("low", "high", "close", "volume"),
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        assert isinstance(source, Tuple) and len(source) == 4
        self.source = source
        self.names = [
            f"cmf_{period}_{self.source[0]}_{self.source[1]}_{self.source[2]}_{self.source[3]}"
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {
                    name: gym.spaces.Box(-1.0, 1.0, shape=(1,))
                    for name in self.names
                }
            )
        else:
            self.spaces = OrderedDict()
        self.money_flow_volumes = []
        self.volumes = []

    def reset(self):
        self.money_flow_volumes.clear()
        self.volumes.clear()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        low, high = getattr(frame, self.source[0]), getattr(
            frame, self.source[1]
        )
        close, volume = getattr(frame, self.source[2]), getattr(
            frame, self.source[3]
        )
        money_flow_multiplier = (
            (close - low - (high - close)) / (high - low)
            if high != low
            else 0.0
        )
        money_flow_volume = volume * money_flow_multiplier
        self.money_flow_volumes.append(money_flow_volume)
        self.volumes.append(volume)
        if len(self.volumes) > self.period:
            self.money_flow_volumes = self.money_flow_volumes[-self.period :]
            self.volumes = self.volumes[-self.period :]
        result = sum(self.money_flow_volumes) / sum(self.volumes)
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result
