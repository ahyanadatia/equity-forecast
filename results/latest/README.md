# results/latest

This directory holds the headline artefacts of the most recent full run — `summary.md` and the
four figures the top-level README embeds. It is the only part of `results/` that is committed;
everything else under `results/` is a per-run output directory and is gitignored.

It is populated by:

```bash
equity-forecast run --config configs/default.yaml
```

which writes `results/<run_id>/` and then copies the summary and figures here.

It is currently empty of results: the session that built this package had no egress to Yahoo
Finance, so no run against real market data has been made yet. Running the command above with
network access will fill it in.
