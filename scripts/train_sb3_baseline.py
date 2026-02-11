#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from intraday.env import SingleAgentEnv
from intraday.providers.dukascopy_local import DukascopyLocalProvider
from intraday.processor import IntervalProcessor
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward
from intraday.features import (
    WILLR, ATR, ROC, CCI, MFI, ADXR, NATR, STDDEV, SMA,
    TickMicrostructure,
)


def make_env(data_dir: str, symbol: str, years: list[int]):
    provider = DukascopyLocalProvider(data_dir=data_dir, symbol=symbol, years=years)
    processor = IntervalProcessor(method="time", interval=300)
    features_pipeline = [
        WILLR(period=14),
        ATR(period=14),
        ROC(period=10),
        CCI(period=20),
        MFI(period=14),
        ADXR(period=14),
        NATR(period=14),
        STDDEV(period=20),
        SMA(period=50),
        TickMicrostructure(),
    ]
    return SingleAgentEnv(
        provider=provider,
        processor=processor,
        action_scheme=BuySellCloseAction(),
        reward_scheme=BalanceReward(),
        features_pipeline=features_pipeline,
        episode_min_duration=3600 * 8,
        episode_max_duration=3600 * 24,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO baseline on Dukascopy tick data")
    parser.add_argument("--data-dir", default="/home/hung/Public/duka-resources")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2012, 2018)))
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--output", default="models/ppo_baseline")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        lambda: make_env(args.data_dir, args.symbol, args.years),
        n_envs=args.n_envs,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/ppo_baseline")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output)
    env.save(args.output + "_vecnorm.pkl")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
