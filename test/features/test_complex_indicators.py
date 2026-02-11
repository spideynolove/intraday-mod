import unittest
from collections import OrderedDict
from intraday.features.cci import CCI
from intraday.features.mfi import MFI
from intraday.features.aroon import Aroon
from intraday.features.natr import NATR
from intraday.frame import Frame


class TestCCI(unittest.TestCase):
    def test_cci_initializes(self):
        cci = CCI(period=20, write_to='state')
        self.assertEqual(cci.names, ['cci_20'])

    def test_cci_calculation(self):
        cci = CCI(period=3, write_to='state')
        cci.reset()

        frames = [
            Frame(high=110, low=90, close=100),
            Frame(high=115, low=95, close=105),
            Frame(high=120, low=100, close=110),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            cci.process(frames[:i+1], state)

        self.assertIsNotNone(state['cci_3'])


class TestMFI(unittest.TestCase):
    def test_mfi_initializes(self):
        mfi = MFI(period=14, write_to='state')
        self.assertEqual(mfi.names, ['mfi_14'])

    def test_mfi_calculation(self):
        mfi = MFI(period=3, write_to='state')
        mfi.reset()

        frames = [
            Frame(high=110, low=90, close=100, volume=1000),
            Frame(high=115, low=95, close=108, volume=1200),
            Frame(high=120, low=100, close=115, volume=1500),
            Frame(high=118, low=98, close=110, volume=1100),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            mfi.process(frames[:i+1], state)

        self.assertGreaterEqual(state['mfi_3'], 0)
        self.assertLessEqual(state['mfi_3'], 100)


class TestAroon(unittest.TestCase):
    def test_aroon_initializes(self):
        aroon = Aroon(period=25, write_to='state')
        self.assertEqual(aroon.names, ['aroon_up_25', 'aroon_down_25'])

    def test_aroon_calculation(self):
        aroon = Aroon(period=3, write_to='state')
        aroon.reset()

        frames = [
            Frame(high=110, low=90),
            Frame(high=115, low=95),
            Frame(high=120, low=100),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            aroon.process(frames[:i+1], state)

        self.assertGreaterEqual(state['aroon_up_3'], 0)
        self.assertLessEqual(state['aroon_up_3'], 100)
        self.assertGreaterEqual(state['aroon_down_3'], 0)
        self.assertLessEqual(state['aroon_down_3'], 100)

    def test_aroon_uptrend(self):
        aroon = Aroon(period=3, write_to='state')
        aroon.reset()

        frames = [
            Frame(high=100, low=90),
            Frame(high=110, low=95),
            Frame(high=120, low=100),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            aroon.process(frames[:i+1], state)

        self.assertEqual(state['aroon_up_3'], 100.0)


class TestNATR(unittest.TestCase):
    def test_natr_initializes(self):
        natr = NATR(period=14, write_to='state')
        self.assertEqual(natr.names, ['natr_14'])

    def test_natr_calculation(self):
        natr = NATR(period=3, write_to='state')
        natr.reset()

        frames = [
            Frame(high=110, low=90, close=100),
            Frame(high=115, low=95, close=105),
            Frame(high=120, low=100, close=110),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            natr.process(frames[:i+1], state)

        self.assertGreater(state['natr_3'], 0)

    def test_natr_is_percentage(self):
        natr = NATR(period=5, write_to='state')
        natr.reset()

        frames = []
        for i in range(10):
            high = 100 + i * 2
            low = 98 + i * 2
            close = 99 + i * 2
            frames.append(Frame(high=high, low=low, close=close))
            state = OrderedDict()
            natr.process(frames, state)

        self.assertGreater(state['natr_5'], 0)
        self.assertLess(state['natr_5'], 100)


if __name__ == '__main__':
    unittest.main()
