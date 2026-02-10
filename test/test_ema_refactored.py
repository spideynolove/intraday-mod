import unittest
from collections import OrderedDict
from intraday.features.ema import EMA
from intraday.frame import Frame


class TestEMARefactored(unittest.TestCase):
    def test_single_source_ema(self):
        ema = EMA(period=10, source='close', write_to='state')
        ema.reset()

        frames = []
        for price in [100, 102, 104, 106, 108]:
            frames.append(Frame(close=price))
            state = OrderedDict()
            ema.process(frames, state)

        self.assertIn('ema_10_close', state)
        self.assertGreater(state['ema_10_close'], 100)
        self.assertLess(state['ema_10_close'], 108)

    def test_multiple_sources_ema(self):
        ema = EMA(period=5, source=['close', 'volume'], write_to='state')
        ema.reset()

        frames = [Frame(close=100.0, volume=1000)]
        state = OrderedDict()
        ema.process(frames, state)

        self.assertEqual(state['ema_5_close'], 100.0)
        self.assertEqual(state['ema_5_volume'], 1000)

    def test_ema_warmup_and_smoothing(self):
        ema = EMA(period=3, source='close', write_to='state')
        ema.reset()

        frames = []
        states = []

        for price in [100, 110, 120, 130]:
            frames.append(Frame(close=price))
            state = OrderedDict()
            ema.process(frames, state)
            states.append(state['ema_3_close'])

        self.assertEqual(states[0], 100.0)
        self.assertEqual(states[1], 105.0)

        ema_factor = 2 / (3 + 1)
        expected = 110.0 * (1 - ema_factor) + 130.0 * ema_factor
        self.assertAlmostEqual(states[3], expected, places=5)


if __name__ == '__main__':
    unittest.main()
