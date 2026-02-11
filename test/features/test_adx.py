import unittest
from collections import OrderedDict
from intraday.features.adx import ADX
from intraday.frame import Frame


class TestADX(unittest.TestCase):
    def test_adx_initializes(self):
        adx = ADX(period=14, write_to='state')

        self.assertEqual(adx.names, ['adx', 'plus_di', 'minus_di'])
        self.assertEqual(len(adx.spaces), 3)

    def test_adx_first_values(self):
        adx = ADX(period=5, write_to='state')
        adx.reset()

        frames = [Frame(high=101, low=99, close=100)]
        state = OrderedDict()
        adx.process(frames, state)

        self.assertIn('adx', state)
        self.assertIn('plus_di', state)
        self.assertIn('minus_di', state)

    def test_adx_strong_uptrend(self):
        adx = ADX(period=5, write_to='state')
        adx.reset()

        frames = []
        for i in range(10):
            high = 100 + i * 2
            low = 98 + i * 2
            close = 99 + i * 2
            frames.append(Frame(high=high, low=low, close=close))
            state = OrderedDict()
            adx.process(frames, state)

        self.assertGreater(state['plus_di'], state['minus_di'])
        self.assertGreater(state['adx'], 0)

    def test_adx_strong_downtrend(self):
        adx = ADX(period=5, write_to='state')
        adx.reset()

        frames = []
        for i in range(10):
            high = 100 - i * 2
            low = 98 - i * 2
            close = 99 - i * 2
            frames.append(Frame(high=high, low=low, close=close))
            state = OrderedDict()
            adx.process(frames, state)

        self.assertGreater(state['minus_di'], state['plus_di'])
        self.assertGreater(state['adx'], 0)

    def test_adx_bounded_0_to_100(self):
        adx = ADX(period=5, write_to='state')
        adx.reset()

        frames = []
        for i in range(15):
            high = 100 + i * 3
            low = 98 + i * 3
            close = 99 + i * 3
            frames.append(Frame(high=high, low=low, close=close))
            state = OrderedDict()
            adx.process(frames, state)

        self.assertGreaterEqual(state['adx'], 0)
        self.assertLessEqual(state['adx'], 100)
        self.assertGreaterEqual(state['plus_di'], 0)
        self.assertLessEqual(state['plus_di'], 100)
        self.assertGreaterEqual(state['minus_di'], 0)
        self.assertLessEqual(state['minus_di'], 100)


if __name__ == '__main__':
    unittest.main()
