from typing import Sequence, Literal, List
from collections import OrderedDict
from abc import ABC, abstractmethod
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
