from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class RSI(Feature):
    def __init__(
        self,
        period: int = 14,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.source = source
        self.names = [f'rsi_{period}']
        self._avg_gain = None
        self._avg_loss = None
        self._prev_value = None
        self._n_iter = 0
        self._ema_factor = 1 / period

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, 100, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._avg_gain = None
        self._avg_loss = None
        self._prev_value = None
        self._n_iter = 0
        self._ema_factor = 1 / self.period

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        current_value = getattr(frame, self.source)

        if self._prev_value is None:
            rsi = 50.0
        else:
            change = current_value - self._prev_value
            gain = max(0, change)
            loss = max(0, -change)

            if self._avg_gain is None:
                self._avg_gain = gain
                self._avg_loss = loss
            elif self._n_iter < self.period:
                self._avg_gain = (self._avg_gain * (self._n_iter - 1) + gain) / self._n_iter
                self._avg_loss = (self._avg_loss * (self._n_iter - 1) + loss) / self._n_iter
            else:
                self._avg_gain = self._avg_gain * (1 - self._ema_factor) + gain * self._ema_factor
                self._avg_loss = self._avg_loss * (1 - self._ema_factor) + loss * self._ema_factor

            if self._avg_loss == 0:
                rsi = 100.0
            elif self._avg_gain == 0:
                rsi = 0.0
            else:
                rs = self._avg_gain / self._avg_loss
                rsi = 100 - (100 / (1 + rs))

        self._prev_value = current_value
        self._n_iter += 1

        if self.write_to_frame:
            setattr(frame, self.names[0], rsi)
        if self.write_to_state:
            state[self.names[0]] = rsi

    def __repr__(self):
        return f"RSI(period={self.period}, source={self.source}, write_to={self.write_to})"
