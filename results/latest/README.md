# results/latest

The headline artefacts of the most recent full run - `summary.md` and the four figures the
top-level README embeds. Alongside it, `results/control/` holds the positive-control run. These
two directories are the only committed parts of `results/`; every other `results/<run_id>/`
directory is a per-run output and is gitignored.

**The run committed here is `synthetic-efficient`: SIMULATED data, not a market result.** The
session that built this package had no egress to Yahoo Finance, so the reference results were
produced from `scripts/make_synthetic_universe.py`. `summary.md` says so in a banner above its
first number, and the tickers are named `SIMA`-`SIMF` so no output can be mistaken for a real
security.

To replace these with a real study:

```bash
equity-forecast run --config configs/default.yaml
```

which writes `results/<run_id>/` and copies the summary and figures here.
