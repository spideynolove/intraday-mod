import unittest
from collections import OrderedDict
from intraday.features.atr import ATR
from intraday.frame import Frame


class TestATR(unittest.TestCase):
    def test_atr_initializes(self):
        atr = ATR(period=14, write_to='state')

        self.assertEqual(atr.period, 14)
        self.assertEqual(atr.names, ['atr_14'])
        self.assertEqual(len(atr.spaces), 1)

    def test_atr_first_value(self):
        atr = ATR(period=5, write_to='state')
        atr.reset()

        frames = [Frame(high=105, low=95, close=100)]
        state = OrderedDict()
        atr.process(frames, state)

        self.assertEqual(state['atr_5'], 10.0)

    def test_atr_with_gaps(self):
        atr = ATR(period=3, write_to='state')
        atr.reset()

        frames = [
            Frame(high=105, low=95, close=100),
            Frame(high=115, low=105, close=110),
            Frame(high=125, low=115, close=120),
        ]

        for i, frame in enumerate(frames):
            state = OrderedDict()
            atr.process(frames[:i+1], state)

        self.assertGreater(state['atr_3'], 0)

    def test_atr_always_positive(self):
        atr = ATR(period=5, write_to='state')
        atr.reset()

        frames = []
        for i in range(10):
            high = 100 + i * 2
            low = 98 + i * 2
            close = 99 + i * 2
            frames.append(Frame(high=high, low=low, close=close))
            state = OrderedDict()
            atr.process(frames, state)

        self.assertGreater(state['atr_5'], 0)

    def test_atr_increases_with_volatility(self):
        atr = ATR(period=3, write_to='state')
        atr.reset()

        frames_low_vol = []
        for i in range(5):
            frames_low_vol.append(Frame(high=101, low=99, close=100))
            state = OrderedDict()
            atr.process(frames_low_vol, state)
        low_vol_atr = state['atr_3']

        atr.reset()
        frames_high_vol = []
        for i in range(5):
            frames_high_vol.append(Frame(high=110, low=90, close=100))
            state = OrderedDict()
            atr.process(frames_high_vol, state)
        high_vol_atr = state['atr_3']

        self.assertGreater(high_vol_atr, low_vol_atr)


if __name__ == '__main__':
    unittest.main()
