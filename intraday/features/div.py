from typing import Sequence, Tuple, Union, Literal
from collections import OrderedDict
from numbers import Real
import math
import warnings
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class Div(Feature):
    def __init__(
        self,
        source: Union[
            Tuple[str, Union[str, Real]],
            Sequence[Tuple[str, Union[str, Real]]],
        ],
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        super().__init__(write_to=write_to)
        if isinstance(source, Tuple) and len(source) == 2:
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        for name1, name2 in self.source:
            assert isinstance(name1, str) and isinstance(name2, (str, Real))
            self.names.append(f"div_{name1}_{str(name2).replace('.', '_')}")
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
        for i, (name1, name2) in enumerate(self.source):
            value1 = getattr(last_frame, name1)
            value2 = (
                getattr(last_frame, name2) if isinstance(name2, str) else name2
            )
            if (
                not isinstance(value1, Real)
                or not isinstance(value2, Real)
                or value2 == 0
            ):
                warnings.warn(
                    f"{self.__class__.__name__}: Invalid value: {name1}={value1}/{name2}={value2}"
                )
                result = 0.0
            else:
                result = value1 / value2
            if self.write_to_frame:
                setattr(last_frame, self.names[i], result)
            if self.write_to_state:
                state[self.names[i]] = result

    def __repr__(self):
        return f"{self.__class__.__name__}(source={self.source}, write_to={self.write_to})"
