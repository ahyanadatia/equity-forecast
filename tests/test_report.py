"""Tests for the reporting layer.

Every test runs against a hand-built stand-in for ``pipeline.RunArtifacts``: the
pipeline itself is never executed, so these tests stay fast, offline and
independent of whatever the models happen to do this week.

The fixture is deliberately a *null* result - no model beats the zero-return
baseline on any measure - because the honesty of the generated findings is the
thing most worth pinning down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from equity_forecast import report
from equity_forecast.backtest import SUMMARY_KEYS, BacktestResult
from equity_forecast.config import (
    BacktestConfig,
    Config,
    DataConfig,
    FeatureConfig,
    ReportConfig,
    RunConfig,
    SplitConfig,
    load_config,
)
from equity_forecast.report import (
    BENCHMARK_FILE,
    FIGURE_FILES,
    HEADLINE_COLUMNS,
    POOLED_TICKER,
    REPORT_FILES,
    build_headline_table,
    generate_findings,
    load_run,
    make_plots,
    render_summary_markdown,
    rerender_report,
    write_report,
)
from equity_forecast.splits import Fold

MODELS = ("zero_return", "ridge", "lstm")
TICKERS = ("SYNA", "SYNB")
N_FOLDS = 4
FOLD_DAYS = 30

# Deliberately hopeless numbers: directional accuracy under the coin flip, DM p-values
# nowhere near 5%, negative net Sharpe for every model.
DIR_ACC = {
    "zero_return": [0.000, 0.000, 0.000, 0.000],
    "ridge": [0.482, 0.474, 0.491, 0.468],
    "lstm": [0.463, 0.497, 0.455, 0.488],
}
IC = {
    "zero_return": [0.0, 0.0, 0.0, 0.0],
    "ridge": [-0.012, 0.004, -0.021, 0.009],
    "lstm": [-0.030, 0.011, -0.006, -0.014],
}
HIT_RATE = {
    "zero_return": [0.5, 0.5, 0.5, 0.5],
    "ridge": [0.481, 0.492, 0.470, 0.488],
    "lstm": [0.466, 0.499, 0.478, 0.461],
}
RMSE = {"zero_return": 0.01200, "ridge": 0.01214, "lstm": 0.01231}
DM_P = {"zero_return": math.nan, "ridge": 0.312, "lstm": 0.674}
DM_STAT = {"zero_return": math.nan, "ridge": 1.02, "lstm": 0.43}
NET_SHARPE = {"zero_return": 0.0, "ridge": -0.34, "lstm": -0.51}
GROSS_SHARPE = {"zero_return": 0.0, "ridge": 0.12, "lstm": -0.09}
TURNOVER = {"zero_return": 0.0, "ridge": 0.88, "lstm": 0.94}


@dataclass(frozen=True)
class FakeArtifacts:
    """Structural stand-in for ``equity_forecast.pipeline.RunArtifacts``."""

    run_id: str
    config: Config
    folds: list[Fold]
    fold_table: pd.DataFrame
    forecasts: pd.DataFrame
    metrics_by_fold: pd.DataFrame
    metrics_summary: pd.DataFrame
    backtests: dict[str, BacktestResult]
    backtest_summary: pd.DataFrame
    pnl: pd.DataFrame
    buy_and_hold: pd.DataFrame


def _config(tmp_path: Path) -> Config:
    return Config(
        run=RunConfig(seed=42, run_id="unittest"),
        data=DataConfig(
            tickers=TICKERS, start="2015-01-01", end="2017-12-31", cache_dir="tests/fixtures/prices"
        ),
        features=FeatureConfig(horizon=1, n_lags=5, roll_windows=(5, 21)),
        splits=SplitConfig(mode="expanding", initial_train_years=1.0, test_months=6, gap_days=1),
        backtest=BacktestConfig(cost_bps=5.0, threshold=0.0, ann_factor=252),
        report=ReportConfig(
            output_dir=str(tmp_path / "results"),
            latest_dir=str(tmp_path / "results" / "latest"),
            make_plots=True,
        ),
    )


def _dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2016-01-04", periods=N_FOLDS * FOLD_DAYS, name="date")


def _folds() -> tuple[list[Fold], pd.DataFrame]:
    dates = _dates()
    folds: list[Fold] = []
    for fold_id in range(N_FOLDS):
        block = dates[fold_id * FOLD_DAYS : (fold_id + 1) * FOLD_DAYS]
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=dates[0],
                train_end=block[0] - pd.Timedelta(days=2),
                test_start=block[0],
                test_end=block[-1],
            )
        )
    table = pd.DataFrame(
        [
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_train_days": 100 + 10 * fold.fold_id,
                "n_test_days": FOLD_DAYS,
            }
            for fold in folds
        ]
    )
    return folds, table


def _forecasts() -> pd.DataFrame:
    rng = np.random.default_rng(20240508)
    dates = _dates()
    rows: list[pd.DataFrame] = []
    truth = pd.DataFrame(
        rng.normal(0.0, 0.012, size=(len(dates), len(TICKERS))), index=dates, columns=list(TICKERS)
    )
    for model in MODELS:
        if model == "zero_return":
            values = np.zeros((len(dates), len(TICKERS)))
        else:
            values = rng.normal(0.0, 0.004, size=(len(dates), len(TICKERS)))
        for j, ticker in enumerate(TICKERS):
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": ticker,
                        "fold_id": np.repeat(np.arange(N_FOLDS), FOLD_DAYS),
                        "model": model,
                        "forecast": values[:, j],
                        "vol_forecast": np.nan,
                        "y_true": truth[ticker].to_numpy(),
                        "ret_1d": truth[ticker].to_numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _metrics_by_fold() -> pd.DataFrame:
    """Mirror the pipeline: per-ticker rows plus a pooled ``ticker == "ALL"`` row."""
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for fold_id in range(N_FOLDS):
            for offset, ticker in enumerate((*TICKERS, POOLED_TICKER)):
                # The pooled row is the fold's headline number; per-ticker rows straddle it.
                shift = 0.0 if ticker == POOLED_TICKER else 0.004 * (offset - 0.5)
                rows.append(
                    {
                        "model": model,
                        "fold_id": fold_id,
                        "ticker": ticker,
                        "n_obs": FOLD_DAYS * (len(TICKERS) if ticker == POOLED_TICKER else 1),
                        "rmse": RMSE[model] + shift * 0.01,
                        "mae": 0.8 * RMSE[model],
                        "dir_acc": DIR_ACC[model][fold_id] + shift,
                        "ic": IC[model][fold_id] + shift,
                        "hit_rate_vs_baseline": HIT_RATE[model][fold_id] + shift,
                    }
                )
    return pd.DataFrame(rows)


def _metrics_summary(by_fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pooled = by_fold[by_fold["ticker"] == POOLED_TICKER]
    for model in MODELS:
        subset = pooled[pooled["model"] == model]
        per_fold = subset.groupby("fold_id")[["rmse", "dir_acc", "ic"]].mean()
        row: dict[str, Any] = {"model": model}
        for metric in ("rmse", "dir_acc", "ic"):
            row[f"{metric}_mean"] = float(per_fold[metric].mean())
            row[f"{metric}_std"] = float(per_fold[metric].std(ddof=1))
        row["n_folds"] = N_FOLDS
        row["dm_stat_vs_zero"] = DM_STAT[model]
        row["dm_p_vs_zero"] = DM_P[model]
        rows.append(row)
    return pd.DataFrame(rows)


def _portfolio(model: str, rng: np.random.Generator) -> pd.DataFrame:
    dates = _dates()
    gross = rng.normal(-0.00005, 0.006, size=len(dates))
    turnover = np.full(len(dates), TURNOVER[model])
    cost = turnover * 5.0 / 1e4
    return pd.DataFrame(
        {"gross": gross, "net": gross - cost, "cost": cost, "turnover": turnover}, index=dates
    )


def _backtests() -> tuple[dict[str, BacktestResult], pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    dates = _dates()
    results: dict[str, BacktestResult] = {}
    summary_rows: list[dict[str, Any]] = []
    pnl_blocks: list[pd.DataFrame] = []
    for model in MODELS:
        portfolio = _portfolio(model, rng)
        positions = pd.DataFrame(
            np.sign(rng.normal(size=(len(dates), len(TICKERS)))), index=dates, columns=list(TICKERS)
        )
        per_ticker_pnl = pd.DataFrame(
            rng.normal(0.0, 0.006, size=(len(dates), len(TICKERS))),
            index=dates,
            columns=list(TICKERS),
        )
        net_summary = dict.fromkeys(SUMMARY_KEYS, 0.0)
        net_summary.update(
            {
                "sharpe": NET_SHARPE[model],
                "ann_return": -0.03,
                "ann_vol": 0.09,
                "max_drawdown": 0.18 + 0.01 * MODELS.index(model),
                "avg_turnover": TURNOVER[model],
                "n_days": float(len(dates)),
            }
        )
        gross_summary = dict(net_summary)
        gross_summary["sharpe"] = GROSS_SHARPE[model]
        results[model] = BacktestResult(
            positions=positions,
            gross_pnl=per_ticker_pnl,
            net_pnl=per_ticker_pnl,
            portfolio=portfolio,
            summary=net_summary,
            per_ticker=pd.DataFrame(
                [net_summary] * len(TICKERS), index=pd.Index(TICKERS, name="ticker")
            ),
            gross_summary=gross_summary,
        )
        summary_rows.append(
            {
                "model": model,
                "gross_sharpe": GROSS_SHARPE[model],
                "net_sharpe": NET_SHARPE[model],
                "gross_ann_return": 0.01,
                "net_ann_return": -0.03,
                "net_max_drawdown": net_summary["max_drawdown"],
                "net_avg_turnover": TURNOVER[model],
                "net_n_days": float(len(dates)),
            }
        )
        pnl_blocks.append(portfolio.reset_index().assign(model=model))
    pnl = pd.concat(pnl_blocks, ignore_index=True)[
        ["date", "model", "gross", "net", "cost", "turnover"]
    ]
    return results, pd.DataFrame(summary_rows), pnl


def _buy_and_hold() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    dates = _dates()
    gross = rng.normal(0.0003, 0.009, size=len(dates))
    turnover = np.zeros(len(dates))
    turnover[0] = 1.0
    return pd.DataFrame(
        {"gross": gross, "net": gross, "cost": np.zeros(len(dates)), "turnover": turnover},
        index=dates,
    )


def make_artifacts(tmp_path: Path) -> FakeArtifacts:
    """Build a complete, internally consistent null-result run."""
    folds, fold_table = _folds()
    by_fold = _metrics_by_fold()
    backtests, backtest_summary, pnl = _backtests()
    return FakeArtifacts(
        run_id="unittest",
        config=_config(tmp_path),
        folds=folds,
        fold_table=fold_table,
        forecasts=_forecasts(),
        metrics_by_fold=by_fold,
        metrics_summary=_metrics_summary(by_fold),
        backtests=backtests,
        backtest_summary=backtest_summary,
        pnl=pnl,
        buy_and_hold=_buy_and_hold(),
    )


@pytest.fixture
def artifacts(tmp_path: Path) -> FakeArtifacts:
    return make_artifacts(tmp_path)


# --------------------------------------------------------------------------- #
# write_report
# --------------------------------------------------------------------------- #
def test_write_report_creates_every_promised_file(artifacts: FakeArtifacts, tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    returned = write_report(artifacts, out_dir=out_dir, copy_to_latest=False)

    assert returned == out_dir
    assert returned.is_dir()
    for name in REPORT_FILES:
        path = out_dir / name
        assert path.exists(), f"{name} was not written"
        assert path.stat().st_size > 0, f"{name} is empty"
    for name in FIGURE_FILES:
        assert (out_dir / name).exists(), f"{name} was not written"


def test_write_report_defaults_to_output_dir_slash_run_id(artifacts: FakeArtifacts) -> None:
    returned = write_report(artifacts, copy_to_latest=False)
    assert returned == Path(artifacts.config.report.output_dir) / artifacts.run_id


def test_written_tables_round_trip(artifacts: FakeArtifacts, tmp_path: Path) -> None:
    out_dir = write_report(artifacts, out_dir=tmp_path / "run", copy_to_latest=False)

    by_fold = pd.read_csv(out_dir / "metrics_by_fold.csv")
    assert len(by_fold) == len(artifacts.metrics_by_fold)
    assert set(by_fold["model"]) == set(MODELS)

    forecasts = pd.read_parquet(out_dir / "forecasts.parquet")
    assert len(forecasts) == len(artifacts.forecasts)
    pnl = pd.read_parquet(out_dir / "pnl.parquet")
    assert set(pnl["model"]) == set(MODELS)


def test_written_config_round_trips_through_load_config(
    artifacts: FakeArtifacts, tmp_path: Path
) -> None:
    out_dir = write_report(artifacts, out_dir=tmp_path / "run", copy_to_latest=False)
    reloaded = load_config(out_dir / "config.yaml")
    assert reloaded == artifacts.config


# --------------------------------------------------------------------------- #
# summary.md
# --------------------------------------------------------------------------- #
def test_summary_md_contains_headline_table_and_every_model(
    artifacts: FakeArtifacts, tmp_path: Path
) -> None:
    out_dir = write_report(artifacts, out_dir=tmp_path / "run", copy_to_latest=False)
    text = (out_dir / "summary.md").read_text(encoding="utf-8")

    for column in HEADLINE_COLUMNS:
        assert column in text, f"headline column {column!r} missing from summary.md"
    for model in MODELS:
        assert model in text, f"model {model!r} missing from summary.md"

    # Ordering required by FR-R3.
    assert text.index("## Run metadata") < text.index("## Headline") < text.index("## Findings")
    assert text.index("## Findings") < text.index("## Walk-forward folds")
    assert (
        text.index("## Walk-forward folds") < text.index("## Backtest") < text.index("## Figures")
    )

    assert artifacts.run_id in text
    assert "SYNA" in text and "5.0 bps" in text and "Seed" in text
    for name in FIGURE_FILES:
        assert f"]({name})" in text, f"figure {name} not embedded in summary.md"


def test_summary_markdown_omits_figures_when_none_were_drawn(artifacts: FakeArtifacts) -> None:
    text = render_summary_markdown(artifacts, figures=[])
    assert "## Figures" not in text
    assert "![" not in text
    assert "## Headline" in text


# --------------------------------------------------------------------------- #
# headline table
# --------------------------------------------------------------------------- #
def test_build_headline_table_has_one_row_per_model_with_mean_std(
    artifacts: FakeArtifacts,
) -> None:
    table = build_headline_table(artifacts)

    assert list(table.columns) == list(HEADLINE_COLUMNS)
    assert len(table) == len(MODELS)
    assert list(table["Model"]) == list(MODELS)

    for column in ("RMSE (mean±std)", "DirAcc (mean±std)", "IC (mean±std)"):
        for cell in table[column]:
            assert "±" in cell, f"{column} cell {cell!r} carries no dispersion"

    ridge = table.set_index("Model").loc["ridge"]
    assert ridge["DM p vs ZeroReturn"] == "0.312"
    assert ridge["Net Sharpe"] == "-0.34"
    assert ridge["Avg turnover"] == "0.880"
    # The baseline is not tested against itself.
    assert table.set_index("Model").loc["zero_return", "DM p vs ZeroReturn"] == "n/a"


def test_headline_dir_acc_matches_the_pooled_per_fold_numbers(artifacts: FakeArtifacts) -> None:
    table = build_headline_table(artifacts).set_index("Model")
    per_fold = artifacts.metrics_by_fold
    ridge = per_fold[(per_fold["model"] == "ridge") & (per_fold["ticker"] == POOLED_TICKER)]
    values = ridge.set_index("fold_id")["dir_acc"]
    expected = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    assert table.loc["ridge", "DirAcc (mean±std)"] == expected


def test_per_fold_numbers_come_from_the_pooled_rows_not_the_ticker_rows(
    artifacts: FakeArtifacts,
) -> None:
    """The pooled row is the fold's score; averaging it with the per-ticker rows would skew it."""
    findings = " ".join(generate_findings(artifacts))
    pooled_mean = pd.Series(DIR_ACC["ridge"]).mean()
    assert f"(mean {pooled_mean:.3f})" in findings


# --------------------------------------------------------------------------- #
# findings
# --------------------------------------------------------------------------- #
def test_generate_findings_is_honest_about_a_null_result(artifacts: FakeArtifacts) -> None:
    findings = generate_findings(artifacts)

    assert 3 <= len(findings) <= 5
    assert all(isinstance(text, str) and text.strip() for text in findings)

    joined = " ".join(findings).lower()

    # The fixture is a null result; the text must say so plainly.
    assert "no model" in joined
    assert "not distinguishable" in joined
    assert "does not survive the cost drag" in joined
    assert "nothing here beats the random walk" in joined

    # ... and must never dress it up as a win.
    for claim in (
        "clear the 5% level",
        "beats the baseline",
        "outperforms",
        "positive net sharpe",
    ):
        assert claim not in joined, f"null result claimed a win: {claim!r}"


def test_findings_quote_the_real_fold_counts_and_p_values(artifacts: FakeArtifacts) -> None:
    joined = " ".join(generate_findings(artifacts))

    # No fold has ridge or lstm above 0.50 in this fixture.
    assert "ridge above 0.50 in 0/4 folds" in joined
    assert "lstm above 0.50 in 0/4 folds" in joined
    # Both DM p-values are reported verbatim, and the smallest is named.
    assert "ridge p=0.312" in joined
    assert "lstm p=0.674" in joined
    assert "0.312 (ridge)" in joined


def test_findings_quantify_the_cost_drag(artifacts: FakeArtifacts) -> None:
    joined = " ".join(generate_findings(artifacts))
    expected_bps = 0.88 * 5.0 * 252
    assert "gross 0.12" in joined
    assert f"{expected_bps:.0f} bps of return per year" in joined
    assert "buy-and-hold over the same window scores" in joined


def test_findings_survive_a_run_with_no_metrics(tmp_path: Path) -> None:
    empty = pd.DataFrame()
    artifacts = make_artifacts(tmp_path)
    stripped = FakeArtifacts(
        run_id=artifacts.run_id,
        config=artifacts.config,
        folds=artifacts.folds,
        fold_table=artifacts.fold_table,
        forecasts=artifacts.forecasts,
        metrics_by_fold=empty,
        metrics_summary=empty,
        backtests={},
        backtest_summary=empty,
        pnl=empty,
        buy_and_hold=empty,
    )
    findings = generate_findings(stripped)
    assert 3 <= len(findings) <= 5
    assert all(text.strip() for text in findings)
    # With nothing measured, the verdict must not pretend to be a verdict.
    joined = " ".join(findings).lower()
    assert "supports no conclusion either way" in joined
    assert "nothing here beats the random walk" not in joined


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def test_make_plots_writes_four_non_trivial_pngs(artifacts: FakeArtifacts, tmp_path: Path) -> None:
    out_dir = tmp_path / "figs"
    paths = make_plots(artifacts, out_dir)

    assert [path.name for path in paths] == list(FIGURE_FILES)
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 5_000, f"{path.name} looks like an empty canvas"
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_make_plots_closes_every_figure(artifacts: FakeArtifacts, tmp_path: Path) -> None:
    plt.close("all")
    make_plots(artifacts, tmp_path / "figs")
    assert plt.get_fignums() == []


def test_make_plots_tolerates_a_run_with_no_pnl(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path)
    empty = pd.DataFrame()
    stripped = FakeArtifacts(
        run_id=artifacts.run_id,
        config=artifacts.config,
        folds=artifacts.folds,
        fold_table=artifacts.fold_table,
        forecasts=empty,
        metrics_by_fold=empty,
        metrics_summary=empty,
        backtests={},
        backtest_summary=empty,
        pnl=empty,
        buy_and_hold=empty,
    )
    paths = make_plots(stripped, tmp_path / "figs")
    assert len(paths) == len(FIGURE_FILES)
    assert all(path.stat().st_size > 1_000 for path in paths)


# --------------------------------------------------------------------------- #
# latest/
# --------------------------------------------------------------------------- #
def test_copy_to_latest_mirrors_summary_and_figures(
    artifacts: FakeArtifacts, tmp_path: Path
) -> None:
    latest = Path(artifacts.config.report.latest_dir)
    assert not latest.exists()

    write_report(artifacts, out_dir=tmp_path / "run", copy_to_latest=True)

    assert (latest / "summary.md").exists()
    for name in FIGURE_FILES:
        assert (latest / name).exists(), f"{name} was not mirrored into latest/"
    assert (latest / "metrics_summary.csv").exists()
    original = (tmp_path / "run" / "summary.md").read_text(encoding="utf-8")
    assert (latest / "summary.md").read_text(encoding="utf-8") == original


def test_copy_to_latest_false_leaves_latest_alone(artifacts: FakeArtifacts, tmp_path: Path) -> None:
    write_report(artifacts, out_dir=tmp_path / "run", copy_to_latest=False)
    assert not Path(artifacts.config.report.latest_dir).exists()


# --------------------------------------------------------------------------- #
# re-rendering from disk
# --------------------------------------------------------------------------- #
def test_rerender_report_rebuilds_the_same_document_from_disk(
    artifacts: FakeArtifacts, tmp_path: Path
) -> None:
    run_dir = write_report(artifacts, out_dir=tmp_path / "run", copy_to_latest=False)
    assert (run_dir / BENCHMARK_FILE).exists()
    original = (run_dir / "summary.md").read_text(encoding="utf-8")

    (run_dir / "summary.md").unlink()
    for name in FIGURE_FILES:
        (run_dir / name).unlink()

    returned = rerender_report(run_dir)

    assert returned == run_dir
    for name in FIGURE_FILES:
        assert (run_dir / name).exists(), f"{name} was not redrawn"
    assert (run_dir / "summary.md").read_text(encoding="utf-8") == original


def test_load_run_reads_back_what_write_report_wrote(
    artifacts: FakeArtifacts, tmp_path: Path
) -> None:
    run_dir = write_report(artifacts, out_dir=tmp_path / "run", copy_to_latest=False)
    loaded = load_run(run_dir)

    assert loaded.run_id == artifacts.run_id
    assert loaded.config == artifacts.config
    assert len(loaded.folds) == len(artifacts.folds)
    assert loaded.folds[0].test_start == artifacts.folds[0].test_start
    assert len(loaded.forecasts) == len(artifacts.forecasts)
    assert set(loaded.metrics_summary["model"]) == set(MODELS)
    # BacktestResult objects are not serialised; the summary table carries their statistics.
    assert loaded.backtests == {}
    assert build_headline_table(loaded).equals(build_headline_table(artifacts))


def test_load_run_rejects_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "does-not-exist")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"config\.yaml"):
        load_run(empty)


# --------------------------------------------------------------------------- #
# Regression: the direction of a Diebold-Mariano result must never be dropped.
#
# A p-value below 5% says the two error series DIFFER. It does not say the model
# won. These tests pin the two ways the report previously turned a null - or
# actively bad - result into a win.
# --------------------------------------------------------------------------- #
def _with_dm(
    artifacts: FakeArtifacts,
    stats: dict[str, float],
    p_values: dict[str, float],
    sharpes: dict[str, float] | None = None,
) -> FakeArtifacts:
    """Return a copy of ``artifacts`` with the DM columns and Sharpes overridden."""
    summary = artifacts.metrics_summary.copy()
    summary["dm_stat_vs_zero"] = summary["model"].map(stats)
    summary["dm_p_vs_zero"] = summary["model"].map(p_values)
    backtest = artifacts.backtest_summary.copy()
    if sharpes is not None:
        backtest["net_sharpe"] = backtest["model"].map(sharpes)
    return replace(artifacts, metrics_summary=summary, backtest_summary=backtest)


def test_significantly_worse_model_is_never_described_as_clearing_the_level(
    artifacts: FakeArtifacts,
) -> None:
    # lstm is significant (p ~ 0) but its POSITIVE statistic means it is worse
    # than the baseline, which is exactly the shipped smoke-run situation.
    worse = _with_dm(
        artifacts,
        stats={"zero_return": math.nan, "ridge": 1.02, "lstm": 9.97},
        p_values={"zero_return": math.nan, "ridge": 0.312, "lstm": 1.8e-21},
    )
    findings = " ".join(report.generate_findings(worse))

    assert "WORSE" in findings
    assert "lstm" in findings
    for win_phrase in ("lstm clear", "lstm beat", "lstm and", "beats the baseline"):
        assert win_phrase not in findings, (
            f"a significantly worse model was written up as {win_phrase!r}"
        )


def test_significance_direction_changes_the_prose(artifacts: FakeArtifacts) -> None:
    # Identical p-values, opposite statistics, must not produce identical text.
    p_values = {"zero_return": math.nan, "ridge": 0.312, "lstm": 0.001}
    better = report.generate_findings(
        _with_dm(artifacts, {"zero_return": math.nan, "ridge": 1.02, "lstm": -9.97}, p_values)
    )
    worse = report.generate_findings(
        _with_dm(artifacts, {"zero_return": math.nan, "ridge": 1.02, "lstm": 9.97}, p_values)
    )
    assert better != worse, "flipping the DM sign left the findings unchanged"
    assert "WORSE" in " ".join(worse)
    assert "WORSE" not in " ".join(better)


def test_bottom_line_does_not_splice_significance_and_profit_across_models(
    artifacts: FakeArtifacts,
) -> None:
    # ridge is significantly BETTER but loses money; lstm makes money but is not
    # significant. Neither model delivers an edge, so the bottom line must not
    # manufacture one by combining them.
    spliced = _with_dm(
        artifacts,
        stats={"zero_return": math.nan, "ridge": -2.60, "lstm": 0.43},
        p_values={"zero_return": math.nan, "ridge": 0.009, "lstm": 0.674},
        sharpes={"zero_return": 0.0, "ridge": -0.34, "lstm": 0.91},
    )
    bottom = report.generate_findings(spliced)[-1]

    assert "cannot be traded profitably" in bottom
    assert "0.91" not in bottom, "quoted another model's Sharpe as though it were the edge"


def test_bottom_line_claims_an_edge_only_when_one_model_delivers_both(
    artifacts: FakeArtifacts,
) -> None:
    vindicated = _with_dm(
        artifacts,
        stats={"zero_return": math.nan, "ridge": -2.60, "lstm": 0.43},
        p_values={"zero_return": math.nan, "ridge": 0.009, "lstm": 0.674},
        sharpes={"zero_return": 0.0, "ridge": 0.77, "lstm": -0.51},
    )
    bottom = report.generate_findings(vindicated)[-1]

    assert "ridge both beats" in bottom
    assert "0.77" in bottom


def test_profitable_but_insignificant_is_reported_as_unproven(artifacts: FakeArtifacts) -> None:
    lucky = _with_dm(
        artifacts,
        stats={"zero_return": math.nan, "ridge": 1.02, "lstm": 0.43},
        p_values={"zero_return": math.nan, "ridge": 0.312, "lstm": 0.674},
        sharpes={"zero_return": 0.0, "ridge": 0.90, "lstm": -0.51},
    )
    bottom = report.generate_findings(lucky)[-1]

    assert "not supported by a measurable accuracy edge" in bottom
    assert "drift or luck, not skill" in bottom


def test_dm_heatmap_series_are_one_observation_per_date(artifacts: FakeArtifacts) -> None:
    # Pooling every (date, ticker) cell would treat correlated same-day errors as
    # independent and inflate significance; the heatmap must use the same
    # cross-sectional design as the headline test.
    series = report._model_error_series(artifacts)
    n_dates = artifacts.forecasts["date"].nunique()

    assert series, "no error series built"
    for name, values in series.items():
        assert len(values) == n_dates, (
            f"{name}: {len(values)} observations for {n_dates} dates - "
            "the heatmap is pooling across tickers"
        )


def test_directional_accuracy_plot_omits_the_baseline(
    artifacts: FakeArtifacts, tmp_path: Path
) -> None:
    """The baseline's structural 0.00 must not set the axis for everyone else.

    zero_return takes no direction, so it scores 0.00 in every fold. Including it
    stretches the y-axis over [0, 0.6] and compresses every model that does take
    a view into the top fifth of the figure, hiding whether the boxes straddle
    0.50 - which is the entire point of the plot.
    """
    fig, ax = plt.subplots()
    try:
        report._plot_dir_acc_box(artifacts, fig, ax)
        labels = [text.get_text() for text in ax.get_xticklabels()]
        bottom, top = ax.get_ylim()
    finally:
        plt.close(fig)

    assert "zero_return" not in labels
    assert set(labels) == {"ridge", "lstm"}
    assert bottom > 0.2, "the axis is still being dragged down by a zero-scoring series"
    assert bottom < 0.5 < top, "the coin-flip line must be visible inside the axis"
