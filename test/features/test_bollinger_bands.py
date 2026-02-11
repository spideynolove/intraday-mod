import unittest
from collections import OrderedDict
from intraday.features.bollinger_bands import BollingerBands
from intraday.frame import Frame


class TestBollingerBands(unittest.TestCase):
    def test_bollinger_bands_initializes(self):
        bb = BollingerBands(period=20, std_dev=2.0, source='close', write_to='state')

        self.assertEqual(bb.names, ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width'])
        self.assertEqual(len(bb.spaces), 4)

    def test_bollinger_bands_first_value(self):
        bb = BollingerBands(period=5, std_dev=2.0, source='close', write_to='state')
        bb.reset()

        frames = [Frame(close=100.0)]
        state = OrderedDict()
        bb.process(frames, state)

        self.assertEqual(state['bb_middle'], 100.0)
        self.assertEqual(state['bb_upper'], 100.0)
        self.assertEqual(state['bb_lower'], 100.0)
        self.assertEqual(state['bb_width'], 0.0)

    def test_bollinger_bands_calculates_bands(self):
        bb = BollingerBands(period=5, std_dev=2.0, source='close', write_to='state')
        bb.reset()

        frames = []
        prices = [100, 102, 98, 103, 97, 105, 95]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            bb.process(frames, state)

        self.assertGreater(state['bb_upper'], state['bb_middle'])
        self.assertLess(state['bb_lower'], state['bb_middle'])
        self.assertGreater(state['bb_width'], 0)

    def test_bollinger_bands_width(self):
        bb = BollingerBands(period=5, std_dev=2.0, source='close', write_to='state')
        bb.reset()

        frames = []
        prices = [100, 102, 98, 103, 97]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            bb.process(frames, state)

        expected_width = state['bb_upper'] - state['bb_lower']
        self.assertAlmostEqual(state['bb_width'], expected_width, places=10)

    def test_bollinger_bands_low_volatility(self):
        bb = BollingerBands(period=5, std_dev=2.0, source='close', write_to='state')
        bb.reset()

        frames = []
        prices = [100, 100, 100, 100, 100]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            bb.process(frames, state)

        self.assertAlmostEqual(state['bb_middle'], 100.0, places=5)
        self.assertAlmostEqual(state['bb_width'], 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
