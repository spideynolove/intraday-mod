from typing import Sequence, Union, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class EMA(Feature):
    def __init__(
        self,
        period: int,
        source: Union[str, Sequence[str]],
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        assert isinstance(period, int) and period > 0
        super().__init__(write_to=write_to, period=period)
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        self._ema_factor = 2 / (self.period + 1)
        for name in self.source:
            setattr(self, name, None)
            self.names.append(f"ema_{self.period}_{name}")
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict(
                {
                    name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                    for name in self.names
                }
            )
        else:
            self.spaces = OrderedDict()
        self._n_iter = 0

    def reset(self):
        self._ema_factor = 2 / (self.period + 1)
        for name in self.source:
            setattr(self, name, None)
        self._n_iter = 0

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            new_value = getattr(last_frame, name)
            new_average = self._update_average_value(name, new_value)
            if self.write_to_frame:
                setattr(last_frame, self.names[i], new_average)
            if self.write_to_state:
                state[self.names[i]] = new_average
        self._n_iter += 1

    def _update_average_value(self, name: str, new_value):
        if new_value is None:
            return
        old_average = getattr(self, name)
        if old_average is None:
            new_average = new_value
        elif self._n_iter < self.period:
            new_average = float(
                old_average * self._n_iter + new_value
            ) / float(self._n_iter + 1)
        else:
            new_average = (
                old_average * (1 - self._ema_factor)
                + new_value * self._ema_factor
            )
        setattr(self, name, new_average)
        return new_average

    def __repr__(self):
        return f"{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})"
