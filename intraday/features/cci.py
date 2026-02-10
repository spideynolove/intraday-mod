from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class CCI(Feature):
    def __init__(
        self,
        period: int = 14,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.names = [f'cci_{period}']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        window_size = min(len(frames), self.period)
        window = frames[-window_size:]

        typical_prices = [(f.high + f.low + f.close) / 3 for f in window]
        sma_tp = sum(typical_prices) / len(typical_prices)
        mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices) / len(typical_prices)

        current_tp = (frames[-1].high + frames[-1].low + frames[-1].close) / 3

        if mean_deviation == 0:
            cci = 0.0
        else:
            cci = (current_tp - sma_tp) / (0.015 * mean_deviation)

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], cci)
        if self.write_to_state:
            state[self.names[0]] = cci
