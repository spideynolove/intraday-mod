from typing import Sequence, Literal
from collections import OrderedDict
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import StatefulEMA


class DEMA(StatefulEMA):
    def __init__(
        self,
        period: int,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(period, source, 'dema', write_to)
        self.names = [f'dema_{period}_{source}']
        self._ema_of_ema = None

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        super().reset()
        self._ema_of_ema = None

    def extract_value(self, frames: Sequence[Frame], source_name: str):
        return getattr(frames[-1], source_name)

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        value = getattr(frame, self.source[0])
        ema = self._update_ema(f'_state_{self.source[0]}', value)

        if ema is not None:
            if self._ema_of_ema is None:
                self._ema_of_ema = ema
            elif self._n_iter < self.period:
                self._ema_of_ema = (self._ema_of_ema * (self._n_iter - 1) + ema) / self._n_iter
            else:
                self._ema_of_ema = self._ema_of_ema * (1 - self._ema_factor) + ema * self._ema_factor

            dema = 2 * ema - self._ema_of_ema
        else:
            dema = None

        self._n_iter += 1

        if self.write_to_frame and dema is not None:
            setattr(frame, self.names[0], dema)
        if self.write_to_state and dema is not None:
            state[self.names[0]] = dema


class TEMA(StatefulEMA):
    def __init__(
        self,
        period: int,
        source: str = 'close',
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(period, source, 'tema', write_to)
        self.names = [f'tema_{period}_{source}']
        self._ema_of_ema = None
        self._ema_of_ema_of_ema = None

        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                self.names[0]: gym.spaces.Box(-math.inf, math.inf, shape=(1,))
            })
        else:
            self.spaces = OrderedDict()

    def reset(self):
        super().reset()
        self._ema_of_ema = None
        self._ema_of_ema_of_ema = None

    def extract_value(self, frames: Sequence[Frame], source_name: str):
        return getattr(frames[-1], source_name)

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        value = getattr(frame, self.source[0])
        ema = self._update_ema(f'_state_{self.source[0]}', value)

        if ema is not None:
            if self._ema_of_ema is None:
                self._ema_of_ema = ema
            elif self._n_iter < self.period:
                self._ema_of_ema = (self._ema_of_ema * (self._n_iter - 1) + ema) / self._n_iter
            else:
                self._ema_of_ema = self._ema_of_ema * (1 - self._ema_factor) + ema * self._ema_factor

            if self._ema_of_ema_of_ema is None:
                self._ema_of_ema_of_ema = self._ema_of_ema
            elif self._n_iter < self.period:
                self._ema_of_ema_of_ema = (self._ema_of_ema_of_ema * (self._n_iter - 1) + self._ema_of_ema) / self._n_iter
            else:
                self._ema_of_ema_of_ema = self._ema_of_ema_of_ema * (1 - self._ema_factor) + self._ema_of_ema * self._ema_factor

            tema = 3 * ema - 3 * self._ema_of_ema + self._ema_of_ema_of_ema
        else:
            tema = None

        self._n_iter += 1

        if self.write_to_frame and tema is not None:
            setattr(frame, self.names[0], tema)
        if self.write_to_state and tema is not None:
            state[self.names[0]] = tema
