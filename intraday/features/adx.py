from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class ADX(Feature):
    def __init__(
        self,
        period: int = 14,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.names = ['adx', 'plus_di', 'minus_di']

        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._atr = None
        self._plus_dm_ema = None
        self._minus_dm_ema = None
        self._adx_value = None
        self._n_iter = 0
        self._ema_factor = 1 / period

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'adx': gym.spaces.Box(0, 100, shape=(1,)),
                'plus_di': gym.spaces.Box(0, 100, shape=(1,)),
                'minus_di': gym.spaces.Box(0, 100, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._atr = None
        self._plus_dm_ema = None
        self._minus_dm_ema = None
        self._adx_value = None
        self._n_iter = 0
        self._ema_factor = 1 / self.period

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]

        if self._prev_high is None:
            self._prev_high = frame.high
            self._prev_low = frame.low
            self._prev_close = frame.close
            plus_di = 0.0
            minus_di = 0.0
            adx = 0.0
        else:
            high_diff = frame.high - self._prev_high
            low_diff = self._prev_low - frame.low

            plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0

            true_range = max(
                frame.high - frame.low,
                abs(frame.high - self._prev_close),
                abs(frame.low - self._prev_close)
            )

            if self._atr is None:
                self._atr = true_range
                self._plus_dm_ema = plus_dm
                self._minus_dm_ema = minus_dm
            else:
                self._atr = self._atr * (1 - self._ema_factor) + true_range * self._ema_factor
                self._plus_dm_ema = self._plus_dm_ema * (1 - self._ema_factor) + plus_dm * self._ema_factor
                self._minus_dm_ema = self._minus_dm_ema * (1 - self._ema_factor) + minus_dm * self._ema_factor

            if self._atr > 0:
                plus_di = 100 * self._plus_dm_ema / self._atr
                minus_di = 100 * self._minus_dm_ema / self._atr
            else:
                plus_di = 0.0
                minus_di = 0.0

            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = 100 * abs(plus_di - minus_di) / di_sum
            else:
                dx = 0.0

            if self._adx_value is None:
                adx = dx
                self._adx_value = dx
            else:
                self._adx_value = self._adx_value * (1 - self._ema_factor) + dx * self._ema_factor
                adx = self._adx_value

            self._prev_high = frame.high
            self._prev_low = frame.low
            self._prev_close = frame.close

        self._n_iter += 1

        if self.write_to_frame:
            setattr(frame, 'adx', adx)
            setattr(frame, 'plus_di', plus_di)
            setattr(frame, 'minus_di', minus_di)
        if self.write_to_state:
            state['adx'] = adx
            state['plus_di'] = plus_di
            state['minus_di'] = minus_di

    def __repr__(self):
        return f"ADX(period={self.period}, write_to={self.write_to})"
