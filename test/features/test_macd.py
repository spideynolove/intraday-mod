import unittest
from collections import OrderedDict
from intraday.features.macd import MACD
from intraday.frame import Frame


class TestMACD(unittest.TestCase):
    def test_macd_initializes(self):
        macd = MACD(fast_period=12, slow_period=26, signal_period=9, source='close', write_to='state')

        self.assertEqual(macd.names, ['macd', 'macd_signal', 'macd_histogram'])
        self.assertEqual(len(macd.spaces), 3)

    def test_macd_first_values_are_zero(self):
        macd = MACD(fast_period=3, slow_period=5, signal_period=3, source='close', write_to='state')
        macd.reset()

        frames = [Frame(close=100.0)]
        state = OrderedDict()
        macd.process(frames, state)

        self.assertEqual(state['macd'], 0.0)
        self.assertEqual(state['macd_signal'], 0.0)
        self.assertEqual(state['macd_histogram'], 0.0)

    def test_macd_uptrend(self):
        macd = MACD(fast_period=3, slow_period=5, signal_period=3, source='close', write_to='state')
        macd.reset()

        frames = []
        prices = [100, 102, 104, 106, 108, 110, 112]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            macd.process(frames, state)

        self.assertGreater(state['macd'], 0)
        self.assertGreater(state['macd_signal'], 0)

    def test_macd_downtrend(self):
        macd = MACD(fast_period=3, slow_period=5, signal_period=3, source='close', write_to='state')
        macd.reset()

        frames = []
        prices = [100, 98, 96, 94, 92, 90, 88]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            macd.process(frames, state)

        self.assertLess(state['macd'], 0)
        self.assertLess(state['macd_signal'], 0)

    def test_macd_histogram_calculation(self):
        macd = MACD(fast_period=3, slow_period=5, signal_period=3, source='close', write_to='state')
        macd.reset()

        frames = []
        prices = [100, 102, 104, 106, 108]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            macd.process(frames, state)

        expected_histogram = state['macd'] - state['macd_signal']
        self.assertAlmostEqual(state['macd_histogram'], expected_histogram, places=10)


if __name__ == '__main__':
    unittest.main()
