from __future__ import annotations
from collections import OrderedDict
from typing import Optional, Sequence
import numpy as np
import gymnasium as gym
from ..feature import Feature
from ..frame import Frame

_NAMES = ["tm_buy_ratio", "tm_vwap_dev", "tm_spread_norm", "tm_spread_exp", "tm_flip_rate"]


class TickMicrostructure(Feature):
    def __init__(self, write_to: str = "state", **kwargs):
        super().__init__(write_to=write_to, **kwargs)
        self.names = _NAMES
        self.spaces = OrderedDict(
            (name, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32))
            for name in _NAMES
        )

    def process(self, frames: Sequence[Frame], state: OrderedDict) -> Optional[OrderedDict]:
        frame = frames[-1]
        ticks = frame.ticks or 1
        buy_ratio = (frame.buy_ticks or 0) / ticks
        vwap = frame.vwap or frame.close
        vwap_dev = (frame.close - vwap) / vwap if vwap else 0.0
        avg_spread = frame.avg_trade_spread or 1e-8
        spread_norm = avg_spread / frame.close if frame.close else 0.0
        spread_exp = (frame.trade_spread_max or avg_spread) / avg_spread
        flip_rate = (frame.flips or 0) / ticks
        if self.write_to_state:
            state["tm_buy_ratio"] = float(buy_ratio)
            state["tm_vwap_dev"] = float(vwap_dev)
            state["tm_spread_norm"] = float(spread_norm)
            state["tm_spread_exp"] = float(spread_exp)
            state["tm_flip_rate"] = float(flip_rate)
        return state
