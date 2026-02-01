# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **modernization fork** of the original `intraday` package - a Gymnasium-compatible environment for simulating intraday trading based on trade streams (historical or real-time).

The codebase simulates realistic exchange trading with order delays, bid-ask spreads, incomplete limit order execution, multi-agent support, and account liquidation on bankruptcy.

## Development Setup

Activate the shared Python environment before any Python operations:
```bash
source /home/hung/env/.venv/bin/activate
```

Install package in editable mode:
```bash
uv pip install -e .
```

Install new dependencies:
```bash
uv pip install <package-name>
```

## Modernization Notes (from setup.py)

Key differences from original package:
- Package name: `intraday-mod` (was `intraday`)
- Author: Hung Nguyen (original: Pavel B. Chernov)
- Repository: `spideynolove/intraday` (original: `diovisgood/intraday`)
- Python: Requires >=3.10 (original: >=3.7)
- Gym: Uses `gymnasium` (original: `gym>=0.17.2`)
- Package manager: Use `uv` (original: `setuptools`)

Dependencies to update/verify:
- numpy (currently >=1.18.1)
- arrow (currently >=0.13.1)
- feather-format (currently >=0.4.1)
- pyglet (optional, currently >=1.5.16)

## Architecture

### Core Components

**Environment Layer** (intraday/env.py):
- `MultiAgentEnv`: Base Gymnasium environment supporting multiple agents trading simultaneously
- `SingleAgentEnv`: Single-agent wrapper with rendering capabilities
- Both use composition: Exchange + Provider + Processor + Features + Actions + Rewards

**Exchange Simulation** (intraday/exchange.py):
- `Exchange`: Core trading simulation with order execution
- Order types: `MarketOrder`, `LimitOrder`, `StopOrder`, `TrailingStopOrder`, `TakeProfitOrder`
- Simulates realistic delays (`agent_order_delay`, `broker_order_delay`)
- Estimates bid-ask spread from trade stream
- Implements `order_luck` (probability of limit order execution when price touches)

**Data Pipeline**:
1. `Provider` (intraday/provider.py): Trade data source
   - `BinanceArchiveProvider`: Downloads monthly trade archives
   - `BinanceKlines`: Downloads candle data (faster for longer intervals)
   - `MoexArchiveProvider`: Moscow Exchange binary format
   - `SineProvider`: Sine wave generator for testing

2. `Processor` (intraday/processor.py): Aggregates trades into frames
   - `IntervalProcessor`: By time, tick count, volume, or money
   - `ImbalanceProcessor`: By buy/sell imbalance
   - `RunProcessor`: By directional runs

3. `Feature` (intraday/feature.py): Extracts features from frames
   - Pipeline of features in `intraday/features/`
   - Many technical indicators: EMA, ADL, CMF, OBV, Fractals, KAMA, Parabolic SAR, etc.

4. `Frame` (intraday/frame.py): OHLCV + metadata container

**Agent Interface**:
- `ActionScheme` (intraday/actions.py): Maps agent actions to orders
  - `BuySellCloseAction`: Discrete {Buy, Sell, Close}
  - `PingPongAction`: Continuous limit order placement

- `RewardScheme` (intraday/rewards.py): Calculates rewards
  - `BalanceReward`: Based on balance change (can normalize by ATR)
  - `ConstantReward`: Fixed values

**Account Management** (intraday/account.py):
- Tracks balance, position, ROI
- Performance metrics via `Report` (intraday/report.py): Sharpe, Sortino, Profit Factor, etc.

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

### Key Design Patterns

**Warm-up Period**: Environment supports `warm_up_time` parameter - trades are processed before `episode_start_datetime` to initialize features without agent actions affecting them.

**Multi-Agent**: Each agent gets its own `Account`. Exchange processes orders for all agents on same trade stream. Useful for evolutionary algorithms.

**Idle Penalty**: Optional `idle_penalty` parameter charges agents for staying out of market (penalty = `idle_penalty * abs(frame.close - frame.open)`).

**Instant vs Delayed Balance Updates**: `instant_balance_update=True` updates unrealized P&L immediately; `False` delays until position closes (more realistic).

**Episode Duration**: Can specify `episode_min_duration` and `episode_max_duration`. Environment auto-closes positions at episode end.

## File Organization

```
intraday/
├── env.py              # Gymnasium environments
├── exchange.py         # Trading simulation
├── account.py          # Account tracking
├── provider.py         # Data provider interface
├── processor.py        # Trade aggregation
├── frame.py            # OHLCV container
├── feature.py          # Feature interface
├── actions.py          # Action schemes
├── rewards.py          # Reward schemes
├── report.py           # Performance metrics
├── render.py           # Visualization
├── simulator.py        # High-level simulator wrapper
├── labels.py           # Labeling utilities
├── providers/          # Specific data providers
└── features/           # Technical indicators
```

## Implementation Guidelines

When modifying this codebase:

1. **Type Annotations**: Use modern type hints (Python 3.10+)
2. **Gymnasium API**: Follow Gymnasium (not old Gym) conventions
3. **Realistic Simulation**: Preserve delay modeling, spread estimation, order luck
4. **Feature Modularity**: New features should inherit from `Feature` base class
5. **Provider Flexibility**: New data sources should implement `Provider` interface
6. **Memory Efficiency**: Environment maintains sliding windows of trades/frames (see `max_trades_period`, `max_frames_period`)

## Legacy Directory

The `legacy/` directory contains the original README with detailed documentation on:
- Installation instructions
- Quick start examples
- Advantages over other trading environments
- Detailed explanations of order delays, spread handling, limit order execution
- Support for multiple agents and idle penalties
- Data provider details
