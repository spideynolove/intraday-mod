import unittest
from collections import OrderedDict
from intraday.features.rsi import RSI
from intraday.frame import Frame


class TestRSI(unittest.TestCase):
    def test_rsi_initializes(self):
        rsi = RSI(period=14, source='close', write_to='state')

        self.assertEqual(rsi.period, 14)
        self.assertEqual(rsi.names, ['rsi_14'])
        self.assertEqual(len(rsi.spaces), 1)

    def test_rsi_first_value_is_50(self):
        rsi = RSI(period=14, source='close', write_to='state')
        rsi.reset()

        frames = [Frame(close=100.0)]
        state = OrderedDict()
        rsi.process(frames, state)

        self.assertEqual(state['rsi_14'], 50.0)

    def test_rsi_all_gains(self):
        rsi = RSI(period=3, source='close', write_to='state')
        rsi.reset()

        frames = []
        prices = [100, 101, 102, 103, 104, 105]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            rsi.process(frames, state)

        self.assertGreater(state['rsi_3'], 90)

    def test_rsi_all_losses(self):
        rsi = RSI(period=3, source='close', write_to='state')
        rsi.reset()

        frames = []
        prices = [100, 99, 98, 97, 96, 95]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            rsi.process(frames, state)

        self.assertLess(state['rsi_3'], 10)

    def test_rsi_mixed_movements(self):
        rsi = RSI(period=5, source='close', write_to='state')
        rsi.reset()

        frames = []
        prices = [100, 102, 101, 103, 102, 104]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            rsi.process(frames, state)

        self.assertGreater(state['rsi_5'], 45)
        self.assertLessEqual(state['rsi_5'], 75)

    def test_rsi_bounded_between_0_and_100(self):
        rsi = RSI(period=3, source='close', write_to='state')
        rsi.reset()

        frames = []
        prices = [100, 110, 120, 130, 140, 150]
        for price in prices:
            frames.append(Frame(close=price))
            state = OrderedDict()
            rsi.process(frames, state)

        self.assertGreaterEqual(state['rsi_3'], 0)
        self.assertLessEqual(state['rsi_3'], 100)


if __name__ == '__main__':
    unittest.main()
