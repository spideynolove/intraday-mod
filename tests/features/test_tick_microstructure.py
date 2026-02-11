import pytest
import numpy as np
from collections import OrderedDict
from unittest.mock import MagicMock
from intraday.features.tick_microstructure import TickMicrostructure


def make_frame(close=1.2, vwap=1.199, ticks=100, buy_ticks=60,
               avg_spread=0.00010, spread_max=0.00020):
    f = MagicMock()
    f.close = close
    f.vwap = vwap
    f.ticks = ticks
    f.buy_ticks = buy_ticks
    f.avg_trade_spread = avg_spread
    f.trade_spread_max = spread_max
    f.flips = 20
    return f


def test_output_shape():
    feat = TickMicrostructure()
    frames = [make_frame() for _ in range(10)]
    state = OrderedDict()
    feat.process(frames, state)
    assert "tick_microstructure" in state
    assert len(state["tick_microstructure"]) == 5


def test_buy_ratio_bounds():
    feat = TickMicrostructure()
    for buy in [0, 50, 100]:
        frames = [make_frame(ticks=100, buy_ticks=buy)]
        state = OrderedDict()
        feat.process(frames, state)
        ratio = state["tick_microstructure"][0]
        assert 0.0 <= ratio <= 1.0


def test_zero_ticks_safe():
    feat = TickMicrostructure()
    frames = [make_frame(ticks=0, buy_ticks=0)]
    state = OrderedDict()
    feat.process(frames, state)
    assert not any(np.isnan(v) for v in state["tick_microstructure"])


def test_vwap_deviation_sign():
    feat = TickMicrostructure()
    frames = [make_frame(close=1.20, vwap=1.19)]
    state = OrderedDict()
    feat.process(frames, state)
    assert state["tick_microstructure"].vwap_deviation > 0


def test_spread_expansion_gte_one():
    feat = TickMicrostructure()
    frames = [make_frame(avg_spread=0.0001, spread_max=0.0003)]
    state = OrderedDict()
    feat.process(frames, state)
    assert state["tick_microstructure"].spread_expansion >= 1.0
