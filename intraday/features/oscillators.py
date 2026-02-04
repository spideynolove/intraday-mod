from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature


class ADXR(Feature):
    def __init__(
        self,
        period: int = 14,
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=period)
        self.names = [f'adxr_{period}']
        self._adx_history = []

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(0, 100, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._adx_history = []

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if not hasattr(frames[-1], 'adx'):
            adxr = 0.0
        else:
            current_adx = frames[-1].adx
            self._adx_history.append(current_adx)

            if len(self._adx_history) > self.period:
                self._adx_history.pop(0)

            if len(self._adx_history) >= self.period:
                adxr = (current_adx + self._adx_history[0]) / 2
            else:
                adxr = current_adx

        if self.write_to_frame:
            setattr(frames[-1], self.names[0], adxr)
        if self.write_to_state:
            state[self.names[0]] = adxr


class APO(Feature):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.source = source
        self.names = ['apo']

        self._fast_ema = None
        self._slow_ema = None
        self._fast_factor = 2 / (fast_period + 1)
        self._slow_factor = 2 / (slow_period + 1)
        self._n_iter = 0

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'apo': gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._fast_ema = None
        self._slow_ema = None
        self._n_iter = 0

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        value = getattr(frames[-1], self.source)

        if self._fast_ema is None:
            self._fast_ema = value
        elif self._n_iter < self.fast_period:
            self._fast_ema = (self._fast_ema * self._n_iter + value) / (self._n_iter + 1)
        else:
            self._fast_ema = self._fast_ema * (1 - self._fast_factor) + value * self._fast_factor

        if self._slow_ema is None:
            self._slow_ema = value
        elif self._n_iter < self.slow_period:
            self._slow_ema = (self._slow_ema * self._n_iter + value) / (self._n_iter + 1)
        else:
            self._slow_ema = self._slow_ema * (1 - self._slow_factor) + value * self._slow_factor

        apo = self._fast_ema - self._slow_ema
        self._n_iter += 1

        if self.write_to_frame:
            setattr(frames[-1], 'apo', apo)
        if self.write_to_state:
            state['apo'] = apo


class PPO(Feature):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to, period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.source = source
        self.names = ['ppo']

        self._fast_ema = None
        self._slow_ema = None
        self._fast_factor = 2 / (fast_period + 1)
        self._slow_factor = 2 / (slow_period + 1)
        self._n_iter = 0

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'ppo': gym.spaces.Box(-100, 100, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        self._fast_ema = None
        self._slow_ema = None
        self._n_iter = 0

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        value = getattr(frames[-1], self.source)

        if self._fast_ema is None:
            self._fast_ema = value
        elif self._n_iter < self.fast_period:
            self._fast_ema = (self._fast_ema * self._n_iter + value) / (self._n_iter + 1)
        else:
            self._fast_ema = self._fast_ema * (1 - self._fast_factor) + value * self._fast_factor

        if self._slow_ema is None:
            self._slow_ema = value
        elif self._n_iter < self.slow_period:
            self._slow_ema = (self._slow_ema * self._n_iter + value) / (self._n_iter + 1)
        else:
            self._slow_ema = self._slow_ema * (1 - self._slow_factor) + value * self._slow_factor

        if self._slow_ema != 0:
            ppo = ((self._fast_ema - self._slow_ema) / self._slow_ema) * 100
        else:
            ppo = 0.0

        self._n_iter += 1

        if self.write_to_frame:
            setattr(frames[-1], 'ppo', ppo)
        if self.write_to_state:
            state['ppo'] = ppo


class BOP(Feature):
    def __init__(self, write_to: Literal["state", "frame", "both"] = "state"):
        super().__init__(write_to=write_to)
        self.names = ['bop']

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'bop': gym.spaces.Box(-1, 1, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        range_hl = frame.high - frame.low

        if range_hl != 0:
            bop = (frame.close - frame.open) / range_hl
        else:
            bop = 0.0

        if self.write_to_frame:
            setattr(frame, 'bop', bop)
        if self.write_to_state:
            state['bop'] = bop
