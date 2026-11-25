"""End-to-end tests for the walk-forward driver and the CLI.

Everything here runs against ``configs/quick.yaml``, which points at the
committed synthetic price fixtures and is marked ``offline: true``, so no test
touches the network. The model list is restricted to the three fast models
except in the one ``slow``-marked test that exercises the real ``smoke``
command (which trains an LSTM).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from equity_forecast import cli, pipeline
from equity_forecast.backtest import SUMMARY_KEYS
from equity_forecast.cli import app
from equity_forecast.config import Config, load_config
from equity_forecast.pipeline import (
    BASELINE_MODEL,
    FORECAST_COLUMNS,
    METRIC_COLUMNS,
    PNL_COLUMNS,
    POOLED_TICKER,
    RunArtifacts,
    resolve_models,
    run_experiment,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICK_CONFIG = REPO_ROOT / "configs" / "quick.yaml"
FIXTURE_PRICES = REPO_ROOT / "tests" / "fixtures" / "prices"

FAST_MODELS: list[str] = ["zero_return", "historical_mean", "ridge"]


def _quick_config() -> Config:
    """Load ``configs/quick.yaml`` with paths made absolute and offline forced."""
    cfg = load_config(QUICK_CONFIG)
    return replace(
        cfg,
        data=replace(cfg.data, cache_dir=str(FIXTURE_PRICES), offline=True),
    )


def _quick_config_file(tmp_path: Path) -> Path:
    """Write a copy of the quick config with an absolute cache directory."""
    raw = yaml.safe_load(QUICK_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(FIXTURE_PRICES)
    raw["data"]["offline"] = True
    destination = tmp_path / "quick.yaml"
    destination.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return destination


@pytest.fixture(scope="module")
def cfg() -> Config:
    return _quick_config()


@pytest.fixture(scope="module")
def artifacts(cfg: Config) -> RunArtifacts:
    return run_experiment(cfg, models=FAST_MODELS)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# --------------------------------------------------------------------------- #
# shape of the artefacts
# --------------------------------------------------------------------------- #
def test_artifacts_expose_the_documented_frames(artifacts: RunArtifacts, cfg: Config) -> None:
    assert artifacts.run_id == "smoke"
    assert artifacts.config is cfg
    assert len(artifacts.folds) >= 2
    assert list(artifacts.fold_table["fold_id"]) == [f.fold_id for f in artifacts.folds]
    assert list(artifacts.fold_table.columns) == [
        "fold_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "n_train_days",
        "n_test_days",
    ]

    forecasts = artifacts.forecasts
    assert list(forecasts.columns) == list(FORECAST_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(forecasts["date"])
    assert pd.api.types.is_string_dtype(forecasts["ticker"])
    assert forecasts["fold_id"].dtype == np.int64
    for column in ("forecast", "vol_forecast", "y_true", "ret_1d"):
        assert forecasts[column].dtype == np.float64
    assert set(forecasts["model"]) == set(FAST_MODELS)
    assert not forecasts["forecast"].isna().any()
    # `y` and `ret_1d` coincide at horizon 1, which is what makes the backtest legible.
    assert cfg.features.horizon == 1
    np.testing.assert_allclose(forecasts["y_true"], forecasts["ret_1d"])

    assert list(artifacts.metrics_by_fold.columns) == [
        "model",
        "fold_id",
        "ticker",
        *METRIC_COLUMNS,
    ]
    assert list(artifacts.pnl.columns) == list(PNL_COLUMNS)
    assert list(artifacts.buy_and_hold.columns) == ["gross", "net", "cost", "turnover"]

    assert set(artifacts.backtests) == set(FAST_MODELS)
    assert list(artifacts.backtest_summary["model"]) == FAST_MODELS
    expected = [
        "model",
        *(f"gross_{key}" for key in SUMMARY_KEYS),
        *(f"net_{key}" for key in SUMMARY_KEYS),
    ]
    assert list(artifacts.backtest_summary.columns) == expected

    summary = artifacts.metrics_summary
    assert list(summary["model"]) == FAST_MODELS
    for metric in ("rmse", "mae", "dir_acc", "ic", "hit_rate_vs_baseline"):
        assert f"{metric}_mean" in summary.columns
        assert f"{metric}_std" in summary.columns
    assert set(summary["n_folds"]) == {len(artifacts.folds)}


def test_every_pooled_row_covers_the_whole_cross_section(artifacts: RunArtifacts) -> None:
    by_fold = artifacts.metrics_by_fold
    pooled = by_fold[by_fold["ticker"] == POOLED_TICKER]
    per_ticker = by_fold[by_fold["ticker"] != POOLED_TICKER]

    totals = per_ticker.groupby(["model", "fold_id"])["n_obs"].sum()
    for row in pooled.itertuples():
        assert row.n_obs == totals.loc[(row.model, row.fold_id)]

    n_folds = artifacts.forecasts["fold_id"].nunique()
    assert len(pooled) == len(FAST_MODELS) * n_folds


# --------------------------------------------------------------------------- #
# the end-to-end no-look-ahead check
# --------------------------------------------------------------------------- #
def test_no_forecast_row_escapes_its_folds_test_block(artifacts: RunArtifacts) -> None:
    bounds = artifacts.fold_table.set_index("fold_id")
    merged = artifacts.forecasts.join(bounds[["train_end", "test_start", "test_end"]], on="fold_id")

    # The scored row is dated after the fold's training data ends - no model is ever
    # graded on a day it was fitted on, and the embargo gap makes the inequality strict.
    assert (merged["date"] > merged["train_end"]).all()
    assert (merged["date"] >= merged["test_start"]).all()
    assert (merged["date"] <= merged["test_end"]).all()

    # Test blocks tile the post-initial-train calendar without overlapping, so a
    # (date, ticker) pair is scored exactly once per model.
    per_model = artifacts.forecasts.groupby("model")
    for _, block in per_model:
        assert not block.duplicated(subset=["date", "ticker"]).any()


def _perturbed_cache(tmp_path: Path, cutoff: pd.Timestamp) -> Path:
    """Copy the price fixtures, scaling every price strictly after ``cutoff``."""
    out = tmp_path / "prices"
    out.mkdir()
    rng = np.random.default_rng(0)
    for source in sorted(FIXTURE_PRICES.glob("*.parquet")):
        frame = pd.read_parquet(source)
        future = frame.index > cutoff
        factors = rng.uniform(0.4, 2.5, int(future.sum()))
        for column in ("open", "high", "low", "close"):
            frame.loc[future, column] = frame.loc[future, column].to_numpy() * factors
        frame.to_parquet(out / source.name)
    return out


def test_mutating_the_far_future_cannot_move_an_earlier_folds_forecasts(
    artifacts: RunArtifacts,
    cfg: Config,
    tmp_path: Path,
) -> None:
    """The sharpest look-ahead check available: rerun on a rewritten future.

    Every price strictly after fold 1's test block is multiplied by a random
    factor and the whole experiment is rerun. Folds 0 and 1 read no row after
    that date - not for their features, not for their targets, and not for the
    standardisation statistics, which are fitted on training rows only - so
    their forecasts must come back bitwise identical. A scaler fitted on the
    whole dataset, or a model handed rows past its window, would fail this.
    """
    cutoff = artifacts.folds[1].test_end
    perturbed = replace(
        cfg, data=replace(cfg.data, cache_dir=str(_perturbed_cache(tmp_path, cutoff)))
    )
    rerun = run_experiment(perturbed, models=FAST_MODELS)

    def _slice(frame: pd.DataFrame, folds: tuple[int, ...]) -> pd.DataFrame:
        selected = frame[frame["fold_id"].isin(folds)]
        return selected.sort_values(["model", "date", "ticker"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        _slice(artifacts.forecasts, (0, 1)),
        _slice(rerun.forecasts, (0, 1)),
        check_exact=True,
    )

    # ...and the perturbation really did bite, so the equality above means something.
    last = artifacts.folds[-1].fold_id
    original_tail = _slice(artifacts.forecasts, (last,))
    rerun_tail = _slice(rerun.forecasts, (last,))
    assert not np.allclose(original_tail["ret_1d"], rerun_tail["ret_1d"])
    ridge = original_tail["model"] == "ridge"
    assert not np.allclose(original_tail.loc[ridge, "forecast"], rerun_tail.loc[ridge, "forecast"])


def test_run_experiment_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg: Config,
) -> None:
    workspace = tmp_path / "cwd"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    run_experiment(cfg, models=FAST_MODELS)
    assert list(workspace.iterdir()) == []


# --------------------------------------------------------------------------- #
# the baseline
# --------------------------------------------------------------------------- #
def test_zero_return_forecasts_are_exactly_zero(artifacts: RunArtifacts) -> None:
    baseline = artifacts.forecasts[artifacts.forecasts["model"] == BASELINE_MODEL]
    assert len(baseline) > 0
    assert (baseline["forecast"] == 0.0).all()
    assert baseline["vol_forecast"].isna().all()

    rows = artifacts.metrics_by_fold
    scores = rows[(rows["model"] == BASELINE_MODEL) & (rows["ticker"] == POOLED_TICKER)]
    # A forecast of exactly zero has no sign, so it matches no realised direction: the
    # documented convention in metrics.directional_accuracy scores it 0.0, not ~0.5.
    assert (scores["dir_acc"] == 0.0).all()
    # It also never beats itself, and its errors are the realised returns themselves.
    assert (scores["hit_rate_vs_baseline"] == 0.0).all()
    assert (scores["rmse"] > 0.0).all()

    others = rows[(rows["model"] != BASELINE_MODEL) & (rows["ticker"] == POOLED_TICKER)]
    assert others["dir_acc"].between(0.3, 0.7).all()


def test_dm_p_vs_zero_is_reported_for_every_challenger(artifacts: RunArtifacts) -> None:
    summary = artifacts.metrics_summary.set_index("model")
    challengers = [name for name in FAST_MODELS if name != BASELINE_MODEL]
    assert challengers

    for name in challengers:
        p_value = summary.loc[name, "dm_p_vs_zero"]
        statistic = summary.loc[name, "dm_stat_vs_zero"]
        assert np.isfinite(p_value)
        assert 0.0 <= p_value <= 1.0
        assert np.isfinite(statistic)

    # The baseline cannot be tested against itself.
    assert np.isnan(summary.loc[BASELINE_MODEL, "dm_p_vs_zero"])


def test_headline_dm_uses_one_observation_per_date(artifacts: RunArtifacts) -> None:
    dm = artifacts.dm_table
    n_dates = artifacts.forecasts["date"].nunique()
    n_rows_per_ticker = (
        artifacts.forecasts[artifacts.forecasts["model"] == "ridge"].groupby("ticker").size()
    )

    pooled = dm[(dm["model"] == "ridge") & (dm["ticker"] == POOLED_TICKER)].iloc[0]
    assert pooled["dm_n_obs"] == n_dates

    # Pooling every (date, ticker) error instead would have inflated the sample by the
    # width of the cross-section; the per-ticker tests are the ones that use those rows.
    for ticker, count in n_rows_per_ticker.items():
        row = dm[(dm["model"] == "ridge") & (dm["ticker"] == ticker)].iloc[0]
        assert row["dm_n_obs"] == count
    assert int(n_rows_per_ticker.sum()) > n_dates


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_the_same_seed_reproduces_the_same_metrics(cfg: Config) -> None:
    first = run_experiment(cfg, models=FAST_MODELS)
    second = run_experiment(cfg, models=FAST_MODELS)
    pd.testing.assert_frame_equal(first.metrics_summary, second.metrics_summary)
    pd.testing.assert_frame_equal(first.backtest_summary, second.backtest_summary)
    pd.testing.assert_frame_equal(first.forecasts, second.forecasts)


def test_resolve_models_defaults_to_the_whole_config_and_rejects_typos(cfg: Config) -> None:
    assert resolve_models(cfg, None) == [
        "zero_return",
        "historical_mean",
        "arima",
        "arima_garch",
        "ridge",
        "lstm",
    ]
    assert resolve_models(cfg, ["ridge", "ridge"]) == ["ridge"]
    with pytest.raises(ValueError, match="unknown model"):
        resolve_models(cfg, ["rigde"])


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_cli_run_exits_zero_and_prints_a_summary(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    config_file = _quick_config_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_file),
            "--models",
            ",".join(FAST_MODELS),
            "--run-id",
            "cli-run",
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run cli-run" in result.output
    for name in FAST_MODELS:
        assert name in result.output
    assert "buy & hold benchmark" in result.output


def test_cli_fetch_populates_from_the_cache_offline(runner: CliRunner, tmp_path: Path) -> None:
    config_file = _quick_config_file(tmp_path)
    result = runner.invoke(app, ["fetch", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "SYNA" in result.output
    assert "SYNB" in result.output


def test_cli_report_on_a_missing_run_exits_non_zero(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["report", "--run-id", "does-not-exist", "--results-dir", str(tmp_path)],
    )
    assert result.exit_code == 1


def test_cli_run_with_an_unknown_model_exits_non_zero(runner: CliRunner, tmp_path: Path) -> None:
    config_file = _quick_config_file(tmp_path)
    result = runner.invoke(
        app,
        ["run", "--config", str(config_file), "--models", "not_a_model"],
    )
    assert result.exit_code == 1


def test_cli_run_hands_the_artifacts_to_write_report(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one entry point the report contract fixes: ``write_report(artifacts)``."""
    seen: list[RunArtifacts] = []
    written = tmp_path / "results" / "stub" / "report.md"

    def _write_report(artifacts: RunArtifacts) -> Path:
        seen.append(artifacts)
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text("stub", encoding="utf-8")
        return written

    monkeypatch.setattr(cli, "_find_report", lambda: SimpleNamespace(write_report=_write_report))
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(_quick_config_file(tmp_path)),
            "--models",
            "zero_return,ridge",
            "--run-id",
            "stub",
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    assert seen[0].run_id == "stub"
    assert seen[0].config.report.output_dir == str(tmp_path / "results")
    assert str(written) in result.output


def test_cli_report_round_trips_a_real_run_from_disk(runner: CliRunner, tmp_path: Path) -> None:
    """`run` then `report` against the real report module, with nothing refitted."""
    report_module = pytest.importorskip("equity_forecast.report")
    results = tmp_path / "results"
    first = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(_quick_config_file(tmp_path)),
            "--models",
            ",".join(FAST_MODELS),
            "--run-id",
            "round-trip",
            "--results-dir",
            str(results),
        ],
    )
    assert first.exit_code == 0, first.output
    run_dir = results / "round-trip"
    for name in report_module.REPORT_FILES:
        assert (run_dir / name).is_file(), name
    original = (run_dir / "summary.md").read_text(encoding="utf-8")

    (run_dir / "summary.md").unlink()
    second = runner.invoke(app, ["report", "--run-id", "round-trip", "--results-dir", str(results)])
    assert second.exit_code == 0, second.output
    assert (run_dir / "summary.md").read_text(encoding="utf-8") == original


def test_cli_report_rerenders_a_previous_run(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    monkeypatch.setattr(
        cli,
        "_find_report",
        lambda: SimpleNamespace(rerender_report=lambda directory: Path(directory) / "report.md"),
    )
    result = runner.invoke(app, ["report", "--run-id", "old-run", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert str(run_dir / "report.md") in result.output


def test_cli_report_fails_loudly_on_an_incomplete_run_directory(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no purpose-built entry point, the fallback rebuild must not paper over gaps."""
    (tmp_path / "old-run").mkdir()
    empty_module = SimpleNamespace()
    monkeypatch.setattr(cli, "_find_report", lambda: empty_module)
    result = runner.invoke(app, ["report", "--run-id", "old-run", "--results-dir", str(tmp_path)])
    assert result.exit_code == 1


@pytest.fixture
def _repo_cwd(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(REPO_ROOT)
    yield


@pytest.mark.slow
@pytest.mark.usefixtures("_repo_cwd")
@pytest.mark.slow
@pytest.mark.requires_tf
def test_cli_smoke_runs_every_model_end_to_end(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["smoke", "--results-dir", str(tmp_path / "results")])
    assert result.exit_code == 0, result.output
    assert "run smoke" in result.output
    for name in ("zero_return", "arima", "arima_garch", "ridge", "lstm"):
        assert name in result.output


# --------------------------------------------------------------------------- #
# Regression: the paired cross-sectional DM inputs must cover the same tickers.
# --------------------------------------------------------------------------- #
def test_paired_cross_sectional_rms_uses_a_common_cross_section() -> None:
    """A model with a NaN ticker must not be averaged over a smaller universe.

    Diebold-Mariano compares two loss series observation by observation, so each
    date's two averages have to be taken over the same tickers. If the model
    leaves ticker B unforecast, averaging each panel over "whatever is finite in
    that panel" compares a one-stock mean against a two-stock mean, and the
    differential then reflects cross-sectional composition rather than forecast
    quality.
    """
    dates = pd.bdate_range("2020-01-01", periods=4, name="date")
    model_panel = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01], "B": [np.nan] * 4}, index=dates)
    base_panel = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01], "B": [0.50] * 4}, index=dates)

    model_rms, base_rms = pipeline._paired_cross_sectional_rms(model_panel, base_panel)

    # Ticker B is dropped from BOTH sides, so the two series are identical and
    # the loss differential is exactly zero - the honest answer, since the model
    # supplied no forecast for B at all.
    np.testing.assert_allclose(model_rms, base_rms)
    np.testing.assert_allclose(model_rms, np.full(4, 0.01))

    # The unpaired computation would have compared 0.01 against ~0.354 and
    # declared the model a huge winner on the strength of a missing forecast.
    naive_base = pipeline._cross_sectional_rms(base_panel)
    assert naive_base[0] > model_rms[0] * 10


def test_paired_cross_sectional_rms_keeps_dates_with_no_shared_ticker_as_nan() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3, name="date")
    model_panel = pd.DataFrame({"A": [0.01, np.nan, 0.02]}, index=dates)
    base_panel = pd.DataFrame({"A": [0.03, 0.04, np.nan]}, index=dates)

    model_rms, base_rms = pipeline._paired_cross_sectional_rms(model_panel, base_panel)

    assert np.isfinite(model_rms[0]) and np.isfinite(base_rms[0])
    assert np.isnan(model_rms[1]) and np.isnan(base_rms[1])
    assert np.isnan(model_rms[2]) and np.isnan(base_rms[2])
