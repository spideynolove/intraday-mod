import unittest
from collections import OrderedDict
from intraday.features.roc import ROC, ROCP, ROCR, ROCR100
from intraday.frame import Frame


class TestROC(unittest.TestCase):
    def test_roc_initializes(self):
        roc = ROC(period=10, source='close', write_to='state')
        self.assertEqual(roc.names, ['roc_10_close'])

    def test_roc_calculation(self):
        roc = ROC(period=2, source='close', write_to='state')
        roc.reset()

        frames = [Frame(close=100), Frame(close=105), Frame(close=110)]
        for i in range(len(frames)):
            state = OrderedDict()
            roc.process(frames[:i+1], state)

        self.assertAlmostEqual(state['roc_2_close'], 10.0, places=2)


class TestROCP(unittest.TestCase):
    def test_rocp_calculation(self):
        rocp = ROCP(period=2, source='close', write_to='state')
        rocp.reset()

        frames = [Frame(close=100), Frame(close=105), Frame(close=110)]
        for i in range(len(frames)):
            state = OrderedDict()
            rocp.process(frames[:i+1], state)

        self.assertAlmostEqual(state['rocp_2_close'], 0.1, places=3)


class TestROCR(unittest.TestCase):
    def test_rocr_calculation(self):
        rocr = ROCR(period=2, source='close', write_to='state')
        rocr.reset()

        frames = [Frame(close=100), Frame(close=105), Frame(close=110)]
        for i in range(len(frames)):
            state = OrderedDict()
            rocr.process(frames[:i+1], state)

        self.assertAlmostEqual(state['rocr_2_close'], 1.1, places=3)


class TestROCR100(unittest.TestCase):
    def test_rocr100_calculation(self):
        rocr100 = ROCR100(period=2, source='close', write_to='state')
        rocr100.reset()

        frames = [Frame(close=100), Frame(close=105), Frame(close=110)]
        for i in range(len(frames)):
            state = OrderedDict()
            rocr100.process(frames[:i+1], state)

        self.assertAlmostEqual(state['rocr100_2_close'], 110.0, places=1)


if __name__ == '__main__':
    unittest.main()
