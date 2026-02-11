import unittest
from collections import OrderedDict
from intraday.features.sma import SMA, WMA, TRIMA
from intraday.features.dema import DEMA, TEMA
from intraday.frame import Frame


class TestSMA(unittest.TestCase):
    def test_sma_initializes(self):
        sma = SMA(period=5, source='close', write_to='state')
        self.assertEqual(sma.names, ['sma_5_close'])

    def test_sma_calculation(self):
        sma = SMA(period=3, source='close', write_to='state')
        sma.reset()

        frames = [Frame(close=100), Frame(close=102), Frame(close=104)]
        for i in range(len(frames)):
            state = OrderedDict()
            sma.process(frames[:i+1], state)

        expected = (100 + 102 + 104) / 3
        self.assertAlmostEqual(state['sma_3_close'], expected, places=2)

    def test_sma_rolling_window(self):
        sma = SMA(period=2, source='close', write_to='state')
        sma.reset()

        frames = [Frame(close=100), Frame(close=110), Frame(close=120)]
        for i in range(len(frames)):
            state = OrderedDict()
            sma.process(frames[:i+1], state)

        expected = (110 + 120) / 2
        self.assertAlmostEqual(state['sma_2_close'], expected, places=2)


class TestWMA(unittest.TestCase):
    def test_wma_initializes(self):
        wma = WMA(period=5, source='close', write_to='state')
        self.assertEqual(wma.names, ['wma_5_close'])

    def test_wma_calculation(self):
        wma = WMA(period=3, source='close', write_to='state')
        wma.reset()

        frames = [Frame(close=100), Frame(close=102), Frame(close=104)]
        for i in range(len(frames)):
            state = OrderedDict()
            wma.process(frames[:i+1], state)

        weights_sum = 1 + 2 + 3
        expected = (100 * 1 + 102 * 2 + 104 * 3) / weights_sum
        self.assertAlmostEqual(state['wma_3_close'], expected, places=2)

    def test_wma_emphasizes_recent(self):
        wma = WMA(period=2, source='close', write_to='state')
        sma = SMA(period=2, source='close', write_to='state')
        wma.reset()
        sma.reset()

        frames = [Frame(close=100), Frame(close=120)]
        for i in range(len(frames)):
            state_wma = OrderedDict()
            state_sma = OrderedDict()
            wma.process(frames[:i+1], state_wma)
            sma.process(frames[:i+1], state_sma)

        self.assertGreater(state_wma['wma_2_close'], state_sma['sma_2_close'])


class TestTRIMA(unittest.TestCase):
    def test_trima_initializes(self):
        trima = TRIMA(period=5, source='close', write_to='state')
        self.assertEqual(trima.names, ['trima_5_close'])

    def test_trima_calculation(self):
        trima = TRIMA(period=3, source='close', write_to='state')
        trima.reset()

        frames = [Frame(close=100), Frame(close=102), Frame(close=104)]
        for i in range(len(frames)):
            state = OrderedDict()
            trima.process(frames[:i+1], state)

        self.assertGreater(state['trima_3_close'], 0)


class TestDEMA(unittest.TestCase):
    def test_dema_initializes(self):
        dema = DEMA(period=5, source='close', write_to='state')
        self.assertEqual(dema.names, ['dema_5_close'])

    def test_dema_calculation(self):
        dema = DEMA(period=3, source='close', write_to='state')
        dema.reset()

        frames = []
        for i in range(10):
            frames.append(Frame(close=100 + i))
            state = OrderedDict()
            dema.process(frames, state)

        self.assertGreater(state['dema_3_close'], 100)


class TestTEMA(unittest.TestCase):
    def test_tema_initializes(self):
        tema = TEMA(period=5, source='close', write_to='state')
        self.assertEqual(tema.names, ['tema_5_close'])

    def test_tema_calculation(self):
        tema = TEMA(period=3, source='close', write_to='state')
        tema.reset()

        frames = []
        for i in range(15):
            frames.append(Frame(close=100 + i))
            state = OrderedDict()
            tema.process(frames, state)

        self.assertGreater(state['tema_3_close'], 100)


if __name__ == '__main__':
    unittest.main()
