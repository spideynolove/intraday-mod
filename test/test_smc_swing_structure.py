import unittest
from collections import OrderedDict
from intraday.features.smc_swing_structure import SwingStructure
from intraday.frame import Frame


class TestSwingStructure(unittest.TestCase):
    def test_initializes(self):
        s = SwingStructure(swing_period=3)
        self.assertIn('swing_high_detected', s.names)
        self.assertIn('swing_low_detected', s.names)
        self.assertIn('bos_bullish', s.names)
        self.assertIn('choch_bullish', s.names)

    def test_detects_swing_high(self):
        s = SwingStructure(swing_period=1)
        s.reset()
        frames = [
            Frame(high=100, low=95, close=98),
            Frame(high=110, low=105, close=108),
            Frame(high=105, low=100, close=102),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            s.process(frames[:i+1], state)
        self.assertEqual(state['swing_high_detected'], 1)
        self.assertAlmostEqual(state['swing_high_price'], 110.0)

    def test_detects_swing_low(self):
        s = SwingStructure(swing_period=1)
        s.reset()
        frames = [
            Frame(high=110, low=105, close=108),
            Frame(high=100, low=90, close=92),
            Frame(high=105, low=100, close=102),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            s.process(frames[:i+1], state)
        self.assertEqual(state['swing_low_detected'], 1)
        self.assertAlmostEqual(state['swing_low_price'], 90.0)

    def test_detects_bullish_bos(self):
        s = SwingStructure(swing_period=1)
        s.reset()
        frames = [
            Frame(high=100, low=90, close=95),
            Frame(high=95, low=85, close=88),
            Frame(high=105, low=92, close=103),
            Frame(high=90, low=80, close=83),
            Frame(high=108, low=86, close=106),
            Frame(high=102, low=95, close=98),
        ]
        for i in range(len(frames)):
            state = OrderedDict()
            s.process(frames[:i+1], state)
        self.assertIn('bos_bullish', state)

    def test_reset_clears_state(self):
        s = SwingStructure(swing_period=1)
        frames = [Frame(high=110, low=90, close=100)]
        for i in range(len(frames)):
            s.process(frames[:i+1], OrderedDict())
        s.reset()
        state = OrderedDict()
        s.process([Frame(high=100, low=95, close=98)], state)
        self.assertEqual(state['swing_high_detected'], 0)


if __name__ == '__main__':
    unittest.main()
