from intraday.providers import ForexSBProvider
from intraday.processor import IntervalProcessor
from intraday.features import SwingStructure, PriceZones, OrderBlock, LiquiditySweep, SessionLevels
from intraday.env import SingleAgentEnv
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward
from datetime import date, timedelta

provider = ForexSBProvider(
    data_dir='./data',
    symbol='EURUSD',
    timeframe='M30',
    date_from=date(2024, 1, 1),
    date_to=date(2024, 3, 31),
    source='dukascopy'
)

processor = IntervalProcessor(method='time', interval=30*60)

smc_features = [
    SwingStructure(swing_period=5),
    PriceZones(range_period=50),
    OrderBlock(impulse_threshold=2.0),
    LiquiditySweep(swing_period=5),
    SessionLevels()
]

env = SingleAgentEnv(
    provider=provider,
    processor=processor,
    features_pipeline=smc_features,
    action_scheme=BuySellCloseAction(),
    reward_scheme=BalanceReward(),
    initial_balance=10000,
    warm_up_time=timedelta(hours=24)
)

state = env.reset()
print(f"Environment initialized with {len(state)} state variables:")
for key in sorted(state.keys())[:10]:
    print(f"  {key}: {state[key]}")

done = False
step_count = 0
while not done and step_count < 10:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    step_count += 1
    print(f"Step {step_count}: reward={reward:.2f}, balance={state.get('balance', 0):.2f}")

print(f"\nCompleted {step_count} steps")
print(f"Final balance: {state.get('balance', 0):.2f}")
print(f"ROI: {state.get('roi', 0):.2%}")
