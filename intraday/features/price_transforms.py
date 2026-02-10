from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class AVGPRICE(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = ['avgprice']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'avgprice': gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        avgprice = (frame.open + frame.high + frame.low + frame.close) / 4

        if self.write_to_frame:
            setattr(frame, 'avgprice', avgprice)
        if self.write_to_state:
            state['avgprice'] = avgprice


class MEDPRICE(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = ['medprice']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'medprice': gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        medprice = (frame.high + frame.low) / 2

        if self.write_to_frame:
            setattr(frame, 'medprice', medprice)
        if self.write_to_state:
            state['medprice'] = medprice


class TYPPRICE(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = ['typprice']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'typprice': gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        typprice = (frame.high + frame.low + frame.close) / 3

        if self.write_to_frame:
            setattr(frame, 'typprice', typprice)
        if self.write_to_state:
            state['typprice'] = typprice


class WCLPRICE(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = ['wclprice']
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'wclprice': gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        wclprice = (frame.high + frame.low + frame.close * 2) / 4

        if self.write_to_frame:
            setattr(frame, 'wclprice', wclprice)
        if self.write_to_state:
            state['wclprice'] = wclprice
