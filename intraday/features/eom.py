from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict
from numbers import Real
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class EOM(Feature):
    def __init__(
        self,
        source: Tuple[str, str, str] = ("low", "high", "volume"),
        price_factor: Union[str, Real, None] = None,
        volume_factor: Union[str, Real, None] = None,
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=2)
        assert isinstance(source, Tuple) and len(source) == 3
        self.source = source
        assert volume_factor is None or isinstance(volume_factor, (str, Real))
        self.price_factor = price_factor
        assert volume_factor is None or isinstance(volume_factor, (str, Real))
        self.volume_factor = volume_factor
        self.names = [
            f"eom_{self.source[0]}_{self.source[1]}_{self.source[2]}"
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
        if len(frames) > 1:
            frame, prev_frame = frames[-1], frames[-2]
            low, high = getattr(frame, self.source[0]), getattr(
                frame, self.source[1]
            )
            prev_low, prev_high = getattr(prev_frame, self.source[0]), getattr(
                prev_frame, self.source[1]
            )
            if isinstance(self.price_factor, Real):
                price_factor = self.price_factor
            elif isinstance(self.price_factor, str):
                price_factor = getattr(frame, self.price_factor)
            else:
                price_factor = 1
            volume = getattr(frame, self.source[2])
            if isinstance(self.volume_factor, Real):
                volume_factor = self.volume_factor
            elif isinstance(self.volume_factor, str):
                volume_factor = getattr(frame, self.volume_factor)
            else:
                volume_factor = 1
            distance = (high + low - prev_high - prev_low) / 2
            result = (
                distance
                * (high - low)
                * volume_factor
                / (volume * price_factor * price_factor)
                if volume != 0 and price_factor != 0
                else 0.0
            )
        else:
            result = 0.0
        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result
        return result
