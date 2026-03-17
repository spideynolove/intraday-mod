# intraday-mod

Intraday trading simulation and DRL experimentation framework, based on `intraday` with added Dukascopy tick-data and SB3 workflow integration.

## Current Project Stage

As of the latest commits in this repo, the project is in the **integration-complete / experimentation-ready** stage.

Completed:
- Dukascopy local tick provider (`intraday/providers/dukascopy_local.py`)
- Dukascopy downloader (`intraday/providers/dukascopy_downloader.py`)
- Tick microstructure feature (`intraday/features/tick_microstructure.py`)
- Economic calendar builder script (`scripts/build_calendar.py`)
- SB3 PPO baseline training script (`scripts/train_sb3_baseline.py`)
- Gymnasium adapter compatibility for SB3 flow
- Tests for new providers/features under `test/providers` and `test/features`

Likely next:
- Run/compare training experiments across feature sets
- Validate out-of-sample performance and risk metrics
- Harden packaging/dependencies in `setup.py`

## Repository Layout

- `intraday/`: core environment, providers, features, rewards, actions
- `scripts/`: helper scripts for calendar construction and PPO training
- `test/`: provider and feature tests
- `docs/`: implementation guides, proposals, and plans

## Docs To Read First

- `docs/plans/2026-02-12-dukascopy-drl-pipeline.md`
- `docs/plans/2026-02-12-drl-fx-brainstorm-summary.md`
- `docs/DUKASCOPY_INTEGRATION_PLAN.md`
- `docs/FOREXSB_PROVIDER_GUIDE.md`

## Quick Start (Local)

1. Install dependencies (project-specific; see `setup.py` and your environment tooling).
2. Ensure Dukascopy `.csv.zst` data exists locally.
3. Run baseline training:

```bash
python scripts/train_sb3_baseline.py \
  --data-dir /path/to/duka-resources \
  --symbol EURUSD \
  --years 2012 2013 2014 2015 2016 2017 \
  --timesteps 500000
```

4. Build merged economic calendar data (optional):

```bash
python scripts/build_calendar.py
```
