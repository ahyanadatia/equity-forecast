# results/latest

The headline artefacts of the committed reference run: `summary.md` and the figures the
top-level README embeds.

**The run committed here is `real-market`:** AAPL, JPM, KO, XOM, MSFT and PG, evaluated from
2015-04-08 to 2026-10-07 on split- and dividend-adjusted daily prices from Yahoo Finance. It was
produced on a GitHub-hosted runner by `.github/workflows/real-data.yml`, which runs
`equity-forecast run --config configs/default.yaml`.

Two control runs on simulated data sit alongside it:

- `results/null/`: no signal planted; the pipeline should find nothing, and does.
- `results/control/`: an AR(1) signal planted; the pipeline should find it, and does.

Every other `results/<run_id>/` directory is a per-run output and is gitignored.
