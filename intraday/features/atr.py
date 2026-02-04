from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class ATR(Feature):
    def __init__(
        self,
        period: int = 14,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.names = [f'atr_{period}']
        self._atr = None
        self._prev_close = None
        self._n_iter = 0
        self._ema_factor = 1 / period

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._atr = None
        self._prev_close = None
        self._n_iter = 0
        self._ema_factor = 1 / self.period

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]

        if self._prev_close is None:
            true_range = frame.high - frame.low
        else:
            true_range = max(
                frame.high - frame.low,
                abs(frame.high - self._prev_close),
                abs(frame.low - self._prev_close)
            )

        if self._atr is None:
            self._atr = true_range
        elif self._n_iter < self.period:
            self._atr = (self._atr * self._n_iter + true_range) / (self._n_iter + 1)
        else:
            self._atr = self._atr * (1 - self._ema_factor) + true_range * self._ema_factor

        self._prev_close = frame.close
        self._n_iter += 1

        if self.write_to_frame:
            setattr(frame, self.names[0], self._atr)
        if self.write_to_state:
            state[self.names[0]] = self._atr

    def __repr__(self):
        return f"ATR(period={self.period}, write_to={self.write_to})"
