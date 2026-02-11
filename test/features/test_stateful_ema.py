import unittest
from collections import OrderedDict
from datetime import datetime
from intraday.feature import StatefulEMA
from intraday.frame import Frame


class ConcreteEMA(StatefulEMA):
    def extract_value(self, frames, source_name):
        return getattr(frames[-1], source_name)


class TestStatefulEMA(unittest.TestCase):
    def test_initializes_with_period_and_source(self):
        ema = ConcreteEMA(period=10, source='close', operation_name='ema', write_to='state')

        self.assertEqual(ema.period, 10)
        self.assertEqual(ema.source, ['close'])
        self.assertEqual(ema.names, ['ema_10_close'])
        self.assertEqual(len(ema.spaces), 1)

    def test_handles_multiple_sources(self):
        ema = ConcreteEMA(period=5, source=['close', 'volume'], operation_name='ema', write_to='state')

        self.assertEqual(ema.source, ['close', 'volume'])
        self.assertEqual(ema.names, ['ema_5_close', 'ema_5_volume'])
        self.assertEqual(len(ema.spaces), 2)

    def test_calculates_ema_for_first_value(self):
        ema = ConcreteEMA(period=10, source='close', operation_name='ema', write_to='state')
        ema.reset()

        frame = Frame(close=100.0)
        state = OrderedDict()
        ema.process([frame], state)

        self.assertEqual(state['ema_10_close'], 100.0)

    def test_calculates_simple_average_during_warmup(self):
        ema = ConcreteEMA(period=3, source='close', operation_name='ema', write_to='state')
        ema.reset()

        frames = [Frame(close=100.0)]
        state1 = OrderedDict()
        ema.process(frames, state1)
        self.assertEqual(state1['ema_3_close'], 100.0)

        frames.append(Frame(close=110.0))
        state2 = OrderedDict()
        ema.process(frames, state2)
        self.assertEqual(state2['ema_3_close'], 105.0)

    def test_uses_ema_formula_after_warmup(self):
        ema = ConcreteEMA(period=3, source='close', operation_name='ema', write_to='state')
        ema.reset()

        frames = [Frame(close=100.0)]
        state1 = OrderedDict()
        ema.process(frames, state1)

        frames.append(Frame(close=110.0))
        state2 = OrderedDict()
        ema.process(frames, state2)

        frames.append(Frame(close=120.0))
        state3 = OrderedDict()
        ema.process(frames, state3)

        frames.append(Frame(close=130.0))
        state4 = OrderedDict()
        ema.process(frames, state4)

        ema_factor = 2 / (3 + 1)
        expected = 110.0 * (1 - ema_factor) + 130.0 * ema_factor
        self.assertAlmostEqual(state4['ema_3_close'], expected, places=5)

    def test_reset_clears_state(self):
        ema = ConcreteEMA(period=3, source='close', operation_name='ema', write_to='state')

        frames = [Frame(close=100.0)]
        state = OrderedDict()
        ema.process(frames, state)

        ema.reset()

        frames = [Frame(close=200.0)]
        state = OrderedDict()
        ema.process(frames, state)

        self.assertEqual(state['ema_3_close'], 200.0)

    def test_write_to_frame(self):
        ema = ConcreteEMA(period=3, source='close', operation_name='ema', write_to='frame')
        ema.reset()

        frame = Frame(close=100.0)
        state = OrderedDict()
        ema.process([frame], state)

        self.assertEqual(frame.ema_3_close, 100.0)
        self.assertNotIn('ema_3_close', state)

    def test_write_to_both(self):
        ema = ConcreteEMA(period=3, source='close', operation_name='ema', write_to='both')
        ema.reset()

        frame = Frame(close=100.0)
        state = OrderedDict()
        ema.process([frame], state)

        self.assertEqual(frame.ema_3_close, 100.0)
        self.assertEqual(state['ema_3_close'], 100.0)


if __name__ == '__main__':
    unittest.main()
