from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class _ROCBase(Feature):
    def __init__(
        self,
        period: int,
        source: str,
        operation_name: str,
        write_to: Literal["state", "frame", "both"],
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.operation_name = operation_name
        self.names = [f'{operation_name}_{period}_{source}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        pass

    def calculate(self, current_value, prev_value):
        raise NotImplementedError()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if len(frames) <= self.period:
            result = 0.0
        else:
            current_value = getattr(frames[-1], self.source)
            prev_value = getattr(frames[-self.period - 1], self.source)
            result = self.calculate(current_value, prev_value)

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], result)
        if self.write_to_state:
            state[self.names[0]] = result


class ROC(_ROCBase):
    def __init__(
        self,
        period: int = 10,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(period, source, 'roc', write_to)

    def calculate(self, current_value, prev_value):
        if prev_value == 0:
            return 0.0
        return ((current_value / prev_value) - 1) * 100

    def __repr__(self):
        return f"ROC(period={self.period}, source={self.source}, write_to={self.write_to})"


class ROCP(_ROCBase):
    def __init__(
        self,
        period: int = 10,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(period, source, 'rocp', write_to)

    def calculate(self, current_value, prev_value):
        if prev_value == 0:
            return 0.0
        return (current_value - prev_value) / prev_value

    def __repr__(self):
        return f"ROCP(period={self.period}, source={self.source}, write_to={self.write_to})"


class ROCR(_ROCBase):
    def __init__(
        self,
        period: int = 10,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(period, source, 'rocr', write_to)

    def calculate(self, current_value, prev_value):
        if prev_value == 0:
            return 0.0
        return current_value / prev_value

    def __repr__(self):
        return f"ROCR(period={self.period}, source={self.source}, write_to={self.write_to})"


class ROCR100(_ROCBase):
    def __init__(
        self,
        period: int = 10,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(period, source, 'rocr100', write_to)

    def calculate(self, current_value, prev_value):
        if prev_value == 0:
            return 0.0
        return (current_value / prev_value) * 100

    def __repr__(self):
        return f"ROCR100(period={self.period}, source={self.source}, write_to={self.write_to})"
