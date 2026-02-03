from typing import Sequence, Literal, List, Union
from collections import OrderedDict
from abc import ABC, abstractmethod
import math
import gymnasium as gym
from .frame import Frame
from .processor import Trade


class Feature(ABC):
    def __init__(
        self, write_to: Literal["state", "frame", "both"] = "state", **kwargs
    ):
        assert isinstance(write_to, str) and write_to in {
            "state",
            "frame",
            "both",
        }
        self.write_to = write_to
        self.write_to_frame: bool = write_to in {"frame", "both"}
        self.write_to_state: bool = write_to in {"state", "both"}
        self.period: int = kwargs["period"] if "period" in kwargs else None
        self.names: List[str] = []
        self.spaces: OrderedDict[str, gym.Space] = OrderedDict()

    def reset(self):
        pass

    @abstractmethod
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        raise NotImplementedError()

    def __str__(self):
        return self.__repr__()


class StatefulEMA(Feature):
    def __init__(
        self,
        period: int,
        source: Union[str, Sequence[str]],
        operation_name: str,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = [source] if isinstance(source, str) else list(source)
        self.operation_name = operation_name
        self.names = [f"ema_{period}_{name}" for name in self.source]
        self._ema_factor = 2 / (period + 1)
        self._n_iter = 0

        for name in self.source:
            setattr(self, f"_state_{name}", None)

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                name: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
                for name in self.names
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._ema_factor = 2 / (self.period + 1)
        self._n_iter = 0
        for name in self.source:
            setattr(self, f"_state_{name}", None)

    @abstractmethod
    def extract_value(self, frames: Sequence[Frame], source_name: str):
        raise NotImplementedError()

    def _update_ema(self, state_name: str, new_value):
        if new_value is None:
            return None
        old_average = getattr(self, state_name)
        if old_average is None:
            new_average = new_value
        elif self._n_iter < self.period:
            new_average = (old_average * self._n_iter + new_value) / (self._n_iter + 1)
        else:
            new_average = old_average * (1 - self._ema_factor) + new_value * self._ema_factor
        setattr(self, state_name, new_average)
        return new_average

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            new_value = self.extract_value(frames, name)
            new_average = self._update_ema(f"_state_{name}", new_value)
            if self.write_to_frame:
                setattr(last_frame, self.names[i], new_average)
            if self.write_to_state:
                state[self.names[i]] = new_average
        self._n_iter += 1

    def __repr__(self):
        return f"{self.__class__.__name__}(period={self.period}, source={self.source}, write_to={self.write_to})"


class TradesFeature(Feature):
    def __init__(
        self, write_to: Literal["state", "frame", "both"] = "state", **kwargs
    ):
        super().__init__(write_to=write_to, **kwargs)
        self.trades_period = (
            kwargs["trades_period"] if "trades_period" in kwargs else None
        )

    @abstractmethod
    def update(self, trades: Sequence[Trade]):
        raise NotImplementedError()
