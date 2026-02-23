# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Modernization fork of the `intraday` package — a Gymnasium-compatible environment for simulating intraday trading based on trade streams (historical or real-time). Simulates realistic exchange trading with order delays, bid-ask spreads, incomplete limit order execution, multi-agent support, and account liquidation.

## Development Setup

```bash
source /home/hung/env/.venv/bin/activate
uv pip install -e .
```

## Running Tests

```bash
# all tests
pytest test/

# by subdirectory
pytest test/features/
pytest test/providers/

# single test file
pytest test/features/test_atr.py

# single test function
pytest test/features/test_atr.py::test_atr_basic -v
```

Note: `test/features/test_dukascopy_api.py` requires network access and will fail offline.

## Scripts

- `scripts/train_sb3_baseline.py` — SB3 PPO training on Dukascopy tick data. Uses argparse: `--data-dir`, `--symbol`, `--years`, `--timesteps`, `--n-envs`, `--output`. Contains `GymnasiumAdapter` that bridges the env's legacy step return `(obs, reward, done, frame)` to Gymnasium's `(obs, reward, terminated, truncated, info)`.
- `scripts/build_calendar.py` — Builds economic calendar CSV from local data (2011-2021) + live FXStreet API (2021-2026). Outputs to `data/`.

## Architecture

### Data Flow

```
Provider → [Trade stream] → Processor → [Frame] → Features → [State dict]
                                ↓
                            Exchange (processes orders)
                                ↓
                            Account (updates balance/position)
                                ↓
                        RewardScheme → [Reward]
```

### Core Components

**Environments** (`intraday/env.py`): `MultiAgentEnv(Exchange, gym.Env)` is the base. `SingleAgentEnv` wraps it for single-agent use with rendering. Both compose: Exchange + Provider + Processor + Features + Actions + Rewards.

**Exchange** (`intraday/exchange.py`): Order matching engine. Order types are `namedtuple`s: `MarketOrder`, `LimitOrder`, `StopOrder`, `TrailingStopOrder`, `TakeProfitOrder`. Models `agent_order_delay`, `broker_order_delay`, bid-ask spread estimation, `order_luck`.

**Providers** (`intraday/provider.py` + `intraday/providers/`): Data sources implementing the `Provider` base class. Includes `BinanceArchiveProvider`, `BinanceKlines`, `MoexArchiveProvider`, `SineProvider`, `ForexSBProvider`, `DukascopyLocalProvider`, `DukascopyDownloader`.

**Processors** (`intraday/processor.py`): Aggregate trades into frames — `IntervalProcessor` (by time/tick/volume/money), `ImbalanceProcessor`, `RunProcessor`.

**Features** (`intraday/feature.py` + `intraday/features/`): ~60 technical indicators inheriting from `Feature` ABC. Contract: `reset()` → `process(frames, state)` → writes values to `state: OrderedDict`. `StatefulEMA` is an intermediate ABC for EMA-based features. `TradesFeature` ABC is for tick-level features.

**Actions** (`intraday/actions.py`): `ActionScheme` ABC with `BuySellCloseAction` (discrete) and `PingPongAction` (continuous limit orders).

**Rewards** (`intraday/rewards.py`): `RewardScheme` base with `ConstantReward` and `BalanceReward` (balance delta, optionally ATR-normalized).

### Key Design Patterns

- **Warm-up period**: `warm_up_time` processes trades before episode start to initialize features without agent actions
- **Multi-agent**: Each agent gets its own `Account`; exchange processes all agents on the same trade stream
- **Idle penalty**: Optional charge for staying flat: `idle_penalty * abs(frame.close - frame.open)`
- **Feature state**: Features write to `OrderedDict` state and/or directly to `Frame` via `write_to` parameter
- **Sliding windows**: `max_trades_period` and `max_frames_period` control memory usage

## Modernization Status

- Package renamed `intraday-mod`, uses `gymnasium` instead of `gym`
- Python >=3.10 with modern type hints (`list[int]`, `str | None`)
- `setup.py` still declares `gym>=0.17.2` (stale — actual code uses `gymnasium`)
- No pyproject.toml migration yet
- No linter/formatter configured (`.gitignore` has `.ruff_cache/`)

## Test Patterns

Tests in `test/features/` (26 files) and `test/providers/` (2 files). Mix of `unittest.TestCase` classes (older) and plain `pytest` functions (newer). Feature tests typically use `MagicMock` for `Frame` objects.
