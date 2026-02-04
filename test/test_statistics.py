import unittest
from collections import OrderedDict
from intraday.features.statistics import BETA, CORREL
from intraday.frame import Frame


class TestBETA(unittest.TestCase):
    def test_beta_initializes(self):
        beta = BETA(period=5, source1='close', source2='volume', write_to='state')
        self.assertEqual(beta.names, ['beta_5_close_volume'])

    def test_beta_calculation(self):
        beta = BETA(period=3, source1='close', source2='volume', write_to='state')
        beta.reset()

        frames = [
            Frame(close=100, volume=1000),
            Frame(close=105, volume=1100),
            Frame(close=110, volume=1200),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            beta.process(frames[:i+1], state)

        self.assertIsNotNone(state['beta_3_close_volume'])

    def test_beta_insufficient_data(self):
        beta = BETA(period=5, source1='close', source2='volume', write_to='state')
        beta.reset()

        frames = [Frame(close=100, volume=1000)]
        state = OrderedDict()
        beta.process(frames, state)

        self.assertEqual(state['beta_5_close_volume'], 0.0)

    def test_beta_zero_variance(self):
        beta = BETA(period=3, source1='close', source2='volume', write_to='state')
        beta.reset()

        frames = [
            Frame(close=100, volume=1000),
            Frame(close=100, volume=1100),
            Frame(close=100, volume=1200),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            beta.process(frames[:i+1], state)

        self.assertEqual(state['beta_3_close_volume'], 0.0)


class TestCORREL(unittest.TestCase):
    def test_correl_initializes(self):
        correl = CORREL(period=30, source1='close', source2='volume', write_to='state')
        self.assertEqual(correl.names, ['correl_30_close_volume'])

    def test_correl_calculation(self):
        correl = CORREL(period=3, source1='close', source2='volume', write_to='state')
        correl.reset()

        frames = [
            Frame(close=100, volume=1000),
            Frame(close=105, volume=1100),
            Frame(close=110, volume=1200),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            correl.process(frames[:i+1], state)

        self.assertGreaterEqual(state['correl_3_close_volume'], -1)
        self.assertLessEqual(state['correl_3_close_volume'], 1)

    def test_correl_perfect_positive(self):
        correl = CORREL(period=3, source1='close', source2='volume', write_to='state')
        correl.reset()

        frames = [
            Frame(close=100, volume=1000),
            Frame(close=110, volume=1100),
            Frame(close=120, volume=1200),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            correl.process(frames[:i+1], state)

        self.assertAlmostEqual(state['correl_3_close_volume'], 1.0, places=5)

    def test_correl_insufficient_data(self):
        correl = CORREL(period=5, source1='close', source2='volume', write_to='state')
        correl.reset()

        frames = [Frame(close=100, volume=1000)]
        state = OrderedDict()
        correl.process(frames, state)

        self.assertEqual(state['correl_5_close_volume'], 0.0)


if __name__ == '__main__':
    unittest.main()
