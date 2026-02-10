# ForexSB Provider Guide

## Overview

`ForexSBProvider` downloads and streams historical Forex data from [data.forexsb.com](https://data.forexsb.com/) - a free source with up to 200,000 bars per symbol.

## Features

- **Timeframes**: M1, M5, M15, M30, H1, H4, D1
- **Data Sources**: Dukascopy, FXCM, TrueFX
- **Format**: CSV (automatically downloaded and cached)
- **Cost**: Free, no API key required

## Basic Usage

```python
from intraday.providers import ForexSBProvider
from intraday.processor import IntervalProcessor
from intraday.env import SingleAgentEnv
from datetime import date

provider = ForexSBProvider(
    data_dir='./data',
    symbol='EURUSD',
    timeframe='M30',
    date_from=date(2024, 1, 1),
    date_to=date(2024, 3, 31),
    source='dukascopy'
)

processor = IntervalProcessor(method='time', interval=30*60)

env = SingleAgentEnv(
    provider=provider,
    processor=processor,
    features_pipeline=[],
    initial_balance=10000
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | Required | Directory to cache downloaded CSV files |
| `symbol` | str | Required | Currency pair (e.g., 'EURUSD', 'GBPUSD') |
| `timeframe` | str | 'M30' | M1, M5, M15, M30, H1, H4, D1 |
| `date_from` | date | 12 months ago | Start date for data |
| `date_to` | date | Now | End date for data |
| `source` | str | 'dukascopy' | Data source: dukascopy, fxcm, truefx |

## Data Format

Downloaded CSV files have this structure:

```
date,time,open,high,low,close,volume
2024.01.01,00:00,1.10450,1.10480,1.10420,1.10455,1234
2024.01.01,00:30,1.10455,1.10520,1.10440,1.10510,987
```

## Caching

Files are cached as:
```
{data_dir}/{SYMBOL}_{TIMEFRAME}_{DATERANGE}.csv
```

Example:
```
./data/EURUSD_M30_20240101_20240331.csv
```

If file exists, no download occurs. Delete file to re-download.

## Comparison with Other Providers

| Provider | Timeframes | Cost | Latency | Best For |
|----------|-----------|------|---------|----------|
| **ForexSBProvider** | M1-D1 | FREE | ~5s download | Forex backtesting |
| **BinanceArchiveProvider** | Tick | FREE | ~10s/month | Crypto tick data |
| **DukascopyProvider** | Tick-D1 | FREE | ~30s download | High-resolution forex |
| **SineProvider** | N/A | FREE | Instant | Testing/debugging |

## Example: M15 EUR/USD with SMC Features

```python
from intraday.providers import ForexSBProvider
from intraday.processor import IntervalProcessor
from intraday.features import SwingStructure, PriceZones, OrderBlock
from intraday.env import SingleAgentEnv
from intraday.actions import BuySellCloseAction
from intraday.rewards import BalanceReward
from datetime import date

provider = ForexSBProvider(
    data_dir='./data',
    symbol='EURUSD',
    timeframe='M15',
    date_from=date(2024, 1, 1),
    date_to=date(2024, 12, 31)
)

processor = IntervalProcessor(method='time', interval=15*60)

features = [
    SwingStructure(swing_period=5),
    PriceZones(range_period=50),
    OrderBlock(impulse_threshold=2.0),
]

env = SingleAgentEnv(
    provider=provider,
    processor=processor,
    features_pipeline=features,
    action_scheme=BuySellCloseAction(),
    reward_scheme=BalanceReward(),
    initial_balance=10000
)

state = env.reset()
print("State keys:", list(state.keys()))
```

## Troubleshooting

### Download fails
```python
requests.exceptions.HTTPError: 404
```
**Solution:** Check symbol spelling, timeframe, and source. Not all combinations exist.

### Empty CSV
```python
IndexError: index 0 is out of bounds
```
**Solution:** Date range has no data. Try different dates or symbol.

### Slow first run
**Expected:** First download takes 5-30 seconds depending on date range. Subsequent runs use cached file.

## Data Sources Comparison

| Source | Coverage | Quality | Update Frequency |
|--------|----------|---------|------------------|
| **Dukascopy** | 2000-present | High (bank-grade) | Daily |
| **FXCM** | 2003-present | Medium | Weekly |
| **TrueFX** | 2009-present | High (bank consortium) | Daily |

**Recommendation:** Use `source='dukascopy'` (default) - most reliable and complete.

## Integration with SMC Features

ForexSB data works perfectly with Smart Money Concepts:

```python
from intraday.features import (
    SwingStructure,
    PriceZones,
    OrderBlock,
    LiquiditySweep,
    SessionLevels
)

smc_features = [
    SwingStructure(swing_period=5),
    PriceZones(range_period=50),
    OrderBlock(impulse_threshold=2.0),
    LiquiditySweep(swing_period=5),
    SessionLevels()
]
```

The forex data includes:
- **Buy/Sell volume** (for order flow imbalance)
- **Tick count** (for liquidity detection)
- **Session times** (Asian/London/NY) via datetime

## Performance Notes

| Timeframe | Bars/Year | Download Time | File Size |
|-----------|-----------|---------------|-----------|
| M1 | 525,600 | 30-60s | ~50 MB |
| M5 | 105,120 | 15-30s | ~10 MB |
| M15 | 35,040 | 5-15s | ~3 MB |
| M30 | 17,520 | 3-10s | ~1.5 MB |
| H1 | 8,760 | 2-5s | ~750 KB |
| H4 | 2,190 | 1-3s | ~200 KB |
| D1 | 365 | <1s | ~30 KB |

**Recommendation for RL training:** Start with M30 or H1 for faster iteration, then refine with M15.

## Next Steps

1. Download data: Provider auto-downloads on first use
2. Create environment: Use with any Processor + Features
3. Train agent: Standard Gymnasium interface
4. Backtest: Use historical date ranges
5. Paper trade: Use recent dates + live order simulation

## References

- [ForexSB Historical Data Portal](https://data.forexsb.com/)
- [ForexSB Data Files Documentation](https://forexsb.com/wiki/fsb/manual/data_files)
- [Top Forex Data Sources Comparison](https://newyorkcityservers.com/blog/top-12-sources-to-download-forex-historical-data-free-paid)
