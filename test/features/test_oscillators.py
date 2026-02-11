import unittest
from collections import OrderedDict
from intraday.features.cmo import CMO
from intraday.features.oscillators import ADXR, APO, PPO, BOP
from intraday.frame import Frame


class TestCMO(unittest.TestCase):
    def test_cmo_initializes(self):
        cmo = CMO(period=14, source='close', write_to='state')
        self.assertEqual(cmo.names, ['cmo_14_close'])

    def test_cmo_calculation(self):
        cmo = CMO(period=3, source='close', write_to='state')
        cmo.reset()

        frames = [Frame(close=100), Frame(close=105), Frame(close=110), Frame(close=115)]
        for i in range(len(frames)):
            state = OrderedDict()
            cmo.process(frames[:i+1], state)

        self.assertGreaterEqual(state['cmo_3_close'], -100)
        self.assertLessEqual(state['cmo_3_close'], 100)

    def test_cmo_uptrend(self):
        cmo = CMO(period=3, source='close', write_to='state')
        cmo.reset()

        frames = [Frame(close=100), Frame(close=110), Frame(close=120), Frame(close=130)]
        for i in range(len(frames)):
            state = OrderedDict()
            cmo.process(frames[:i+1], state)

        self.assertGreater(state['cmo_3_close'], 0)


class TestADXR(unittest.TestCase):
    def test_adxr_initializes(self):
        adxr = ADXR(period=14, write_to='state')
        self.assertEqual(adxr.names, ['adxr_14'])

    def test_adxr_calculation(self):
        adxr = ADXR(period=5, write_to='state')
        adxr.reset()

        frames = []
        for i in range(20):
            high = 100 + i + (i % 3)
            low = 95 + i - (i % 2)
            close = 98 + i
            frames.append(Frame(high=high, low=low, close=close))
            state = OrderedDict()
            adxr.process(frames, state)

        self.assertGreaterEqual(state['adxr_5'], 0)
        self.assertLessEqual(state['adxr_5'], 100)


class TestAPO(unittest.TestCase):
    def test_apo_initializes(self):
        apo = APO(fast_period=12, slow_period=26, source='close', write_to='state')
        self.assertEqual(apo.names, ['apo_12_26_close'])

    def test_apo_calculation(self):
        apo = APO(fast_period=3, slow_period=6, source='close', write_to='state')
        apo.reset()

        frames = []
        for i in range(15):
            frames.append(Frame(close=100 + i))
            state = OrderedDict()
            apo.process(frames, state)

        self.assertIsNotNone(state['apo_3_6_close'])


class TestPPO(unittest.TestCase):
    def test_ppo_initializes(self):
        ppo = PPO(fast_period=12, slow_period=26, source='close', write_to='state')
        self.assertEqual(ppo.names, ['ppo_12_26_close'])

    def test_ppo_calculation(self):
        ppo = PPO(fast_period=3, slow_period=6, source='close', write_to='state')
        ppo.reset()

        frames = []
        for i in range(15):
            frames.append(Frame(close=100 + i))
            state = OrderedDict()
            ppo.process(frames, state)

        self.assertIsNotNone(state['ppo_3_6_close'])


class TestBOP(unittest.TestCase):
    def test_bop_initializes(self):
        bop = BOP(write_to='state')
        self.assertEqual(bop.names, ['bop'])

    def test_bop_calculation(self):
        bop = BOP(write_to='state')
        bop.reset()

        frames = [Frame(open=100, high=110, low=95, close=108)]
        state = OrderedDict()
        bop.process(frames, state)

        expected = (108 - 100) / (110 - 95)
        self.assertAlmostEqual(state['bop'], expected, places=3)

    def test_bop_range(self):
        bop = BOP(write_to='state')
        bop.reset()

        frames = [Frame(open=100, high=110, low=95, close=105)]
        state = OrderedDict()
        bop.process(frames, state)

        self.assertGreaterEqual(state['bop'], -1)
        self.assertLessEqual(state['bop'], 1)


if __name__ == '__main__':
    unittest.main()
