import unittest
from collections import OrderedDict
from intraday.features.smc_liquidity_sweep import LiquiditySweep
from intraday.frame import Frame


class TestLiquiditySweep(unittest.TestCase):
    def test_initializes(self):
        ls = LiquiditySweep()
        self.assertIn('liquidity_above', ls.names)
        self.assertIn('sweep_high_detected', ls.names)
        self.assertIn('inducement_bullish', ls.names)

    def test_liquidity_levels_tracked(self):
        ls = LiquiditySweep(swing_period=1)
        ls.reset()
        frames = [
            Frame(high=100, low=90, close=95),
            Frame(high=110, low=105, close=108),
            Frame(high=105, low=100, close=102),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            ls.process(frames[:i+1], state)
        self.assertGreater(state['liquidity_above'], 0)

    def test_sweep_high_detected(self):
        ls = LiquiditySweep(swing_period=1, sweep_reversal_threshold=0.3)
        ls.reset()
        frames = [
            Frame(high=100, low=90, close=95),
            Frame(high=110, low=105, close=108),
            Frame(high=105, low=100, close=102),
            Frame(high=115, low=108, close=109),
            Frame(high=112, low=103, close=104),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            ls.process(frames[:i+1], state)
        self.assertEqual(state['sweep_high_detected'], 1)

    def test_reset_clears_state(self):
        ls = LiquiditySweep()
        frames = [Frame(high=110, low=90, close=100)]
        ls.process(frames, OrderedDict())
        ls.reset()
        state = OrderedDict()
        ls.process([Frame(high=100, low=95, close=98)], state)
        self.assertEqual(state['liquidity_above_count'], 0)


if __name__ == '__main__':
    unittest.main()
