import unittest
from collections import OrderedDict
from intraday.features.trange import TRANGE
from intraday.features.mom import MOM
from intraday.features.stddev import STDDEV, VAR
from intraday.frame import Frame


class TestTRANGE(unittest.TestCase):
    def test_trange_initializes(self):
        trange = TRANGE(write_to='state')
        self.assertEqual(trange.names, ['trange'])

    def test_trange_first_frame(self):
        trange = TRANGE(write_to='state')
        trange.reset()

        frames = [Frame(high=110, low=90, close=100)]
        state = OrderedDict()
        trange.process(frames, state)

        self.assertAlmostEqual(state['trange'], 20.0, places=2)

    def test_trange_with_gaps(self):
        trange = TRANGE(write_to='state')
        trange.reset()

        frames = [
            Frame(high=110, low=90, close=100),
            Frame(high=120, low=85, close=115),
        ]

        for i in range(len(frames)):
            state = OrderedDict()
            trange.process(frames[:i+1], state)

        expected = max(120 - 85, abs(120 - 100), abs(85 - 100))
        self.assertAlmostEqual(state['trange'], expected, places=2)


class TestMOM(unittest.TestCase):
    def test_mom_initializes(self):
        mom = MOM(period=5, source='close', write_to='state')
        self.assertEqual(mom.names, ['mom_5_close'])

    def test_mom_calculation(self):
        mom = MOM(period=3, source='close', write_to='state')
        mom.reset()

        frames = [Frame(close=100), Frame(close=105), Frame(close=110), Frame(close=115)]
        for i in range(len(frames)):
            state = OrderedDict()
            mom.process(frames[:i+1], state)

        self.assertAlmostEqual(state['mom_3_close'], 15.0, places=2)


class TestSTDDEV(unittest.TestCase):
    def test_stddev_initializes(self):
        stddev = STDDEV(period=5, source='close', write_to='state')
        self.assertEqual(stddev.names, ['stddev_5_close'])

    def test_stddev_calculation(self):
        stddev = STDDEV(period=3, source='close', write_to='state')
        stddev.reset()

        frames = [Frame(close=100), Frame(close=102), Frame(close=104)]
        for i in range(len(frames)):
            state = OrderedDict()
            stddev.process(frames[:i+1], state)

        self.assertGreater(state['stddev_3_close'], 0)

    def test_stddev_constant_values(self):
        stddev = STDDEV(period=3, source='close', write_to='state')
        stddev.reset()

        frames = [Frame(close=100), Frame(close=100), Frame(close=100)]
        for i in range(len(frames)):
            state = OrderedDict()
            stddev.process(frames[:i+1], state)

        self.assertAlmostEqual(state['stddev_3_close'], 0.0, places=5)


class TestVAR(unittest.TestCase):
    def test_var_initializes(self):
        var = VAR(period=5, source='close', write_to='state')
        self.assertEqual(var.names, ['var_5_close'])

    def test_var_calculation(self):
        var = VAR(period=3, source='close', write_to='state')
        var.reset()

        frames = [Frame(close=100), Frame(close=102), Frame(close=104)]
        for i in range(len(frames)):
            state = OrderedDict()
            var.process(frames[:i+1], state)

        self.assertGreater(state['var_3_close'], 0)

    def test_var_is_stddev_squared(self):
        period = 3
        stddev = STDDEV(period=period, source='close', write_to='state')
        var = VAR(period=period, source='close', write_to='state')
        stddev.reset()
        var.reset()

        frames = [Frame(close=100), Frame(close=105), Frame(close=110)]
        for i in range(len(frames)):
            state_stddev = OrderedDict()
            state_var = OrderedDict()
            stddev.process(frames[:i+1], state_stddev)
            var.process(frames[:i+1], state_var)

        expected_var = state_stddev['stddev_3_close'] ** 2
        self.assertAlmostEqual(state_var['var_3_close'], expected_var, places=5)


if __name__ == '__main__':
    unittest.main()
