import unittest
from collections import OrderedDict
from intraday.features.willr import WILLR
from intraday.frame import Frame


class TestWILLR(unittest.TestCase):
    def test_willr_initializes(self):
        willr = WILLR(period=14, write_to='state')

        self.assertEqual(willr.period, 14)
        self.assertEqual(willr.names, ['willr_14'])
        self.assertEqual(len(willr.spaces), 1)

    def test_willr_single_frame(self):
        willr = WILLR(period=5, write_to='state')
        willr.reset()

        frames = [Frame(high=105, low=95, close=100)]
        state = OrderedDict()
        willr.process(frames, state)

        self.assertEqual(state['willr_5'], -50.0)

    def test_willr_at_high(self):
        willr = WILLR(period=5, write_to='state')
        willr.reset()

        frames = []
        for i in range(10):
            frames.append(Frame(high=100 + i, low=90 + i, close=100 + i))
            state = OrderedDict()
            willr.process(frames, state)

        self.assertGreater(state['willr_5'], -10)

    def test_willr_at_low(self):
        willr = WILLR(period=5, write_to='state')
        willr.reset()

        frames = []
        for i in range(10):
            frames.append(Frame(high=100 - i, low=90 - i, close=90 - i))
            state = OrderedDict()
            willr.process(frames, state)

        self.assertLess(state['willr_5'], -90)

    def test_willr_bounded_negative_100_to_0(self):
        willr = WILLR(period=5, write_to='state')
        willr.reset()

        frames = []
        for i in range(20):
            high = 100 + (i % 3) * 10
            low = 90 + (i % 3) * 10
            close = 95 + (i % 3) * 10
            frames.append(Frame(high=high, low=low, close=close))
            state = OrderedDict()
            willr.process(frames, state)

        self.assertGreaterEqual(state['willr_5'], -100)
        self.assertLessEqual(state['willr_5'], 0)


if __name__ == '__main__':
    unittest.main()
