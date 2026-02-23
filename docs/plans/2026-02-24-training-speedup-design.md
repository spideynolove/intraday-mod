# Training Speedup: SubprocVecEnv + Shared Memory + Cython Hot Loop

## Problem

Training PPO on GBPUSD tick data is bottlenecked by CPU-bound Python env stepping (~90-95% of wall time). The neural network (2x64 MLP) is tiny; GPU provides negligible benefit. The hot path is `_get_next_frame` which iterates hundreds of ticks per step through pure Python exchange/processor logic.

## Approach

Two complementary optimizations that preserve full tick-level simulation fidelity:

1. **SubprocVecEnv + shared memory** — true multi-process parallelism, zero data duplication
2. **Cython hot loop** — 5-10x speedup on per-tick exchange and processor logic

Combined expected speedup: **N_cores x 5-10x** on env stepping throughput.

## Part 1: SubprocVecEnv + Shared Memory

### Overview

Switch from `DummyVecEnv` (sequential) to `SubprocVecEnv` (parallel). Parent process loads numpy arrays once into `multiprocessing.shared_memory`; child processes attach zero-copy views.

### Memory layout (per year, 150M trades)

| Array | dtype | Size |
|-------|-------|------|
| `_datetimes` | datetime64[us] (8 bytes) | 1.2 GB |
| `_is_buy` | bool (1 byte) | 0.15 GB |
| `_amounts` | float64 (8 bytes) | 1.2 GB |
| `_prices` | float64 (8 bytes) | 1.2 GB |
| **Total** | | **~3.75 GB** |

With shared memory: 6 years = ~22.5 GB total regardless of n_envs.

### API

```python
# Parent process
provider = DukascopyLocalProvider(data_dir=..., symbol=..., years=...)
provider._ensure_loaded()
shm_refs = provider.share_memory()
# Returns: {"datetimes": {"name": str, "dtype": str, "shape": tuple}, ...}

# Child process
provider = DukascopyLocalProvider.from_shared_memory(shm_refs, symbol=..., years=...)
```

### Files changed

- `intraday/providers/dukascopy_local.py` — add `share_memory()`, `from_shared_memory()`, update `close()`
- `scripts/train_sb3_baseline.py` — load in parent, pass `shm_refs` via closure to `make_vec_env`, use `SubprocVecEnv`

### Shared memory lifecycle

- Parent: `share_memory()` creates `SharedMemory` blocks, stores references
- Children: `from_shared_memory()` attaches, creates numpy views (read-only)
- Children `close()`: detach only (no unlink)
- Parent cleanup: `unlink_shared_memory()` frees OS shared memory blocks
- Use `atexit` or `try/finally` in training script to ensure cleanup

## Part 2: Cython Hot Loop

### Target methods

Phase 1 (highest impact):

1. `exchange.py:process_trade()` (lines 178-347) — per-tick order matching
2. `processor.py:IntervalProcessor.process()` (lines 63+) — per-tick frame aggregation

Phase 2 (if Phase 1 validates the approach):

3. `env.py:_get_next_frame()` (lines 326-365) — the outer loop
4. `TickMicrostructure` feature updates

### Cython design

Replace Python namedtuples and dicts with typed C structs and arrays:

```
Trade namedtuple → (double price, int operation, long long dt_us, double amount)
OrderedDict of orders → typed C array with integer type tags
isinstance() dispatch → switch on type tag integer
```

Thin Python wrapper preserves existing API:

```
exchange.py imports FastExchange from _exchange.pyx
Falls back to pure Python Exchange if Cython extension not built
```

### New files

- `intraday/_exchange.pyx` — Cython `cdef class FastExchange`
- `intraday/_processor.pyx` — Cython `cdef class FastIntervalProcessor`
- `setup.py` — add `ext_modules` with Cython build

### Existing files modified

- `intraday/exchange.py` — conditional import from `_exchange`, fallback to pure Python
- `intraday/processor.py` — conditional import from `_processor`, fallback to pure Python

### Build

```bash
pip install cython
python setup.py build_ext --inplace
```

Or via setup.py:

```python
from Cython.Build import cythonize
ext_modules = cythonize([
    "intraday/_exchange.pyx",
    "intraday/_processor.pyx",
])
```

### Fallback strategy

Pure Python always available. Cython is optional:

```python
try:
    from ._exchange import FastExchange as _ExchangeImpl
except ImportError:
    from .exchange import Exchange as _ExchangeImpl
```

## Implementation Order

1. SubprocVecEnv switch (trivial, immediate win with existing envs)
2. Shared memory for DukascopyLocalProvider (medium effort)
3. Cython Phase 1: `process_trade` + `IntervalProcessor.process`
4. Benchmark after each step
5. Cython Phase 2 if Phase 1 shows gains

## Verification

After each step:

```bash
pytest test/ -v
python scripts/train_sb3_baseline.py --timesteps 10000 --n-envs 4  # smoke test
```

Compare wall-clock time per 1000 timesteps before/after each optimization.
