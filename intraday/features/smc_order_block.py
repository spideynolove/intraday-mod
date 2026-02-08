from typing import Sequence, Literal
from collections import OrderedDict, namedtuple
import math
import gymnasium as gym
from intraday.frame import Frame
from intraday.feature import Feature

OBZone = namedtuple("OBZone", "top bottom frame_idx")
FVGZone = namedtuple("FVGZone", "top bottom gap_size frame_idx")


class OrderBlock(Feature):
    def __init__(
        self,
        impulse_threshold: float = 2.0,
        max_blocks: int = 50,
        fvg_threshold: float = 0.0,
        source: tuple[str, str, str, str] = ("open", "high", "low", "close"),
        write_to: Literal["state", "frame", "both"] = "state",
    ):
        super().__init__(write_to=write_to)
        self.impulse_threshold = impulse_threshold
        self.max_blocks = max_blocks
        self.fvg_threshold = fvg_threshold
        self.source = source
        self.names = [
            'bullish_ob_detected',
            'bearish_ob_detected',
            'bullish_ob_top',
            'bullish_ob_bottom',
            'bearish_ob_top',
            'bearish_ob_bottom',
            'in_bullish_ob',
            'in_bearish_ob',
            'fvg_bullish_detected',
            'fvg_bearish_detected',
            'fvg_bullish_gap_size',
            'fvg_bearish_gap_size',
        ]
        if write_to in {"state", "both"}:
            self.spaces = OrderedDict({
                'bullish_ob_detected': gym.spaces.Discrete(2),
                'bearish_ob_detected': gym.spaces.Discrete(2),
                'bullish_ob_top': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bullish_ob_bottom': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bearish_ob_top': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'bearish_ob_bottom': gym.spaces.Box(-math.inf, math.inf, shape=(1,)),
                'in_bullish_ob': gym.spaces.Discrete(2),
                'in_bearish_ob': gym.spaces.Discrete(2),
                'fvg_bullish_detected': gym.spaces.Discrete(2),
                'fvg_bearish_detected': gym.spaces.Discrete(2),
                'fvg_bullish_gap_size': gym.spaces.Box(0, math.inf, shape=(1,)),
                'fvg_bearish_gap_size': gym.spaces.Box(0, math.inf, shape=(1,)),
            })
        else:
            self.spaces = OrderedDict()
        self._bullish_obs: list[OBZone] = []
        self._bearish_obs: list[OBZone] = []
        self._bullish_fvgs: list[FVGZone] = []
        self._bearish_fvgs: list[FVGZone] = []
        self._atr_ema: float | None = None
        self._atr_factor = 2 / 15

    def reset(self):
        self._bullish_obs.clear()
        self._bearish_obs.clear()
        self._bullish_fvgs.clear()
        self._bearish_fvgs.clear()
        self._atr_ema = None

    def _update_atr(self, frame: Frame):
        tr = getattr(frame, 'true_range', abs(frame.high - frame.low))
        if self._atr_ema is None:
            self._atr_ema = tr
        else:
            self._atr_ema = self._atr_ema * (1 - self._atr_factor) + tr * self._atr_factor

    def _evict_mitigated(self, close: float):
        self._bullish_obs = [z for z in self._bullish_obs if close > z.bottom]
        self._bearish_obs = [z for z in self._bearish_obs if close < z.top]

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        open_s, high_s, low_s, close_s = self.source
        frame = frames[-1]
        self._update_atr(frame)
        close = getattr(frame, close_s)
        atr = self._atr_ema if self._atr_ema and self._atr_ema > 0 else 1e-9

        bullish_ob_detected = 0
        bearish_ob_detected = 0
        fvg_bullish_detected = 0
        fvg_bearish_detected = 0
        fvg_bullish_gap = 0.0
        fvg_bearish_gap = 0.0

        if len(frames) >= 2:
            prev = frames[-2]
            prev_close = getattr(prev, close_s)
            move = close - prev_close
            impulse = abs(move) > self.impulse_threshold * atr

            if impulse and move > 0:
                ob_open = getattr(prev, open_s)
                ob_close = getattr(prev, close_s)
                if ob_close < ob_open:
                    top = max(ob_open, ob_close)
                    bottom = min(ob_open, ob_close)
                    self._bullish_obs.append(OBZone(top, bottom, len(frames) - 2))
                    if len(self._bullish_obs) > self.max_blocks:
                        self._bullish_obs.pop(0)
                    bullish_ob_detected = 1

            if impulse and move < 0:
                ob_open = getattr(prev, open_s)
                ob_close = getattr(prev, close_s)
                if ob_close > ob_open:
                    top = max(ob_open, ob_close)
                    bottom = min(ob_open, ob_close)
                    self._bearish_obs.append(OBZone(top, bottom, len(frames) - 2))
                    if len(self._bearish_obs) > self.max_blocks:
                        self._bearish_obs.pop(0)
                    bearish_ob_detected = 1

        if len(frames) >= 3:
            f0, f2 = frames[-3], frames[-1]
            gap_up = getattr(f2, low_s) - getattr(f0, high_s)
            gap_down = getattr(f0, low_s) - getattr(f2, high_s)
            if gap_up > self.fvg_threshold:
                self._bullish_fvgs.append(FVGZone(getattr(f2, low_s), getattr(f0, high_s), gap_up, len(frames) - 1))
                fvg_bullish_detected = 1
                fvg_bullish_gap = gap_up
            elif gap_down > self.fvg_threshold:
                self._bearish_fvgs.append(FVGZone(getattr(f0, low_s), getattr(f2, high_s), gap_down, len(frames) - 1))
                fvg_bearish_detected = 1
                fvg_bearish_gap = gap_down

        self._evict_mitigated(close)

        bullish_ob_top = self._bullish_obs[-1].top if self._bullish_obs else 0.0
        bullish_ob_bottom = self._bullish_obs[-1].bottom if self._bullish_obs else 0.0
        bearish_ob_top = self._bearish_obs[-1].top if self._bearish_obs else 0.0
        bearish_ob_bottom = self._bearish_obs[-1].bottom if self._bearish_obs else 0.0

        in_bullish = int(any(z.bottom <= close <= z.top for z in self._bullish_obs))
        in_bearish = int(any(z.bottom <= close <= z.top for z in self._bearish_obs))

        values = [
            bullish_ob_detected, bearish_ob_detected,
            bullish_ob_top, bullish_ob_bottom, bearish_ob_top, bearish_ob_bottom,
            in_bullish, in_bearish,
            fvg_bullish_detected, fvg_bearish_detected,
            fvg_bullish_gap, fvg_bearish_gap,
        ]

        if self.write_to_frame:
            for name, val in zip(self.names, values):
                setattr(frame, name, val)
        if self.write_to_state:
            for name, val in zip(self.names, values):
                state[name] = val
