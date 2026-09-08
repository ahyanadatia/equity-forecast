"""Command-line interface: ``equity-forecast <command>``.

Four commands:

``run``
    Execute a walk-forward experiment from a YAML config and write its report.
``report``
    Re-render a previous run's report from the artefacts already on disk.
``smoke``
    Run ``configs/quick.yaml`` against the committed fixtures, fully offline.
    This is what CI executes, so it must never touch the network.
``fetch``
    Populate the price cache and stop, so a later run can be fully offline.

Every command logs progress to stdout at ``INFO`` and exits non-zero if
anything raises. Writing artefacts is delegated to
:mod:`equity_forecast.report`; this module only decides *when* to call it.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import pandas as pd
import typer

from equity_forecast.backtest import buy_and_hold, summarise_pnl
from equity_forecast.config import Config, load_config
from equity_forecast.data import download_prices
from equity_forecast.pipeline import RunArtifacts, run_experiment
from equity_forecast.splits import Fold

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

__all__ = ["app", "configure_logging"]

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Walk-forward comparison of return-forecasting models, with a cost-aware backtest.",
)

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
_HANDLER_NAME = "equity-forecast-cli"

_DEFAULT_CONFIG = Path("configs/default.yaml")
_QUICK_CONFIG = Path("configs/quick.yaml")

# `report.py` is owned by another module; `write_report(artifacts) -> Path` is the one entry
# point its contract fixes. Re-rendering a finished run from disk is not in that contract, so
# the `report` command first probes for a purpose-built entry point and otherwise rebuilds the
# artefacts from the files `write_report` itself wrote (see `_load_artifacts`).
# Deliberately unambiguous, directory-shaped names only: a generic name like `render_report`
# would collide with a function that takes artefacts rather than a path.
_RERENDER_ENTRY_POINTS: tuple[str, ...] = ("rerender_report", "write_report_from_dir")
_LOAD_ENTRY_POINTS: tuple[str, ...] = ("load_artifacts",)

_HEADLINE_COLUMNS: tuple[str, ...] = (
    "model",
    "rmse_mean",
    "dir_acc_mean",
    "ic_mean",
    "dm_p_vs_zero",
)


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def configure_logging(level: int = logging.INFO) -> None:
    """Send the package's log records to stdout in a terse format.

    The handler is attached to the ``equity_forecast`` logger rather than the
    root logger, so importing the CLI never hijacks another application's (or
    pytest's) logging configuration. Calling this twice replaces the previous
    handler instead of duplicating output.

    Args:
        level: Level for the package logger; ``logging.INFO`` by default.
    """
    package = logging.getLogger("equity_forecast")
    for existing in list(package.handlers):
        if getattr(existing, "name", None) == _HANDLER_NAME:
            package.removeHandler(existing)
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATEFMT))
    package.addHandler(handler)
    package.setLevel(level)


def _fail(action: str, exc: BaseException) -> NoReturn:
    """Log a failure and exit non-zero.

    Args:
        action: What was being attempted, for the log line.
        exc: The exception that ended it.

    Raises:
        Exit: Always, with code 1.
    """
    logger.error("%s failed: %s", action, exc, exc_info=True)
    raise typer.Exit(code=1) from exc


def _split_models(models: str | None) -> list[str] | None:
    """Parse a ``--models`` value into a list of registry keys.

    Args:
        models: Comma-separated model names, or ``None``.

    Returns:
        The names in the order given, or ``None`` meaning "every configured model".
    """
    if models is None:
        return None
    names = [part.strip() for part in models.split(",")]
    return [name for name in names if name]


def _apply_overrides(
    cfg: Config,
    run_id: str | None = None,
    results_dir: Path | None = None,
) -> Config:
    """Return ``cfg`` with the command-line overrides applied.

    Args:
        cfg: The configuration as loaded from YAML.
        run_id: Overrides ``run.run_id`` when given.
        results_dir: Overrides ``report.output_dir`` (and ``report.latest_dir``,
            which becomes ``<results_dir>/latest``) when given.

    Returns:
        A new frozen :class:`~equity_forecast.config.Config`.
    """
    if run_id:
        cfg = replace(cfg, run=replace(cfg.run, run_id=run_id))
    if results_dir is not None:
        cfg = replace(
            cfg,
            report=replace(
                cfg.report,
                output_dir=str(results_dir),
                latest_dir=str(results_dir / "latest"),
            ),
        )
    return cfg


def _find_report() -> ModuleType | None:
    """Import :mod:`equity_forecast.report` if it exists.

    Returns:
        The imported module, or ``None`` when the module does not exist. A
        module that exists but fails to import raises, so a broken report is
        never mistaken for an absent one.
    """
    if importlib.util.find_spec("equity_forecast.report") is None:
        return None
    return importlib.import_module("equity_forecast.report")


def _write_report(artifacts: RunArtifacts) -> Path | None:
    """Write a run's artefacts to disk.

    Args:
        artifacts: The finished run.

    Returns:
        The directory (or file) the report module reported writing, or ``None``
        when :mod:`equity_forecast.report` does not exist yet. An *absent*
        report module is a packaging gap rather than a modelling failure, so it
        downgrades to a warning; a module that exists but raises still fails the
        command.
    """
    module = _find_report()
    if module is None:
        logger.warning(
            "equity_forecast.report does not exist; run %s produced results but wrote "
            "nothing to disk",
            artifacts.run_id,
        )
        return None
    return Path(module.write_report(artifacts))


def _load_artifacts(run_dir: Path) -> RunArtifacts:
    """Rebuild a :class:`~equity_forecast.pipeline.RunArtifacts` from a run directory.

    ``write_report`` persists every frame the report is rendered from, so a
    finished run can be reconstituted without refitting anything. Two fields are
    not on disk and are recovered rather than stored: the buy-and-hold benchmark
    is recomputed from the saved realised returns, and ``backtests`` (the full
    per-model :class:`~equity_forecast.backtest.BacktestResult` objects) comes
    back empty - the report reads its statistics from ``backtest_summary.csv``,
    which is saved, and only consults ``backtests`` as a fallback.

    Args:
        run_dir: Directory a previous run wrote into.

    Returns:
        Artefacts good enough to re-render the whole report.
    """
    cfg = load_config(run_dir / "config.yaml")
    date_columns = ["train_start", "train_end", "test_start", "test_end"]
    fold_table = pd.read_csv(run_dir / "folds.csv", parse_dates=date_columns)
    boundaries = [pd.DatetimeIndex(fold_table[column]) for column in date_columns]
    folds = [
        Fold(
            fold_id=int(fold_id),
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        for fold_id, train_start, train_end, test_start, test_end in zip(
            fold_table["fold_id"].astype("int64"), *boundaries, strict=True
        )
    ]
    forecasts = pd.read_parquet(run_dir / "forecasts.parquet")
    unique = forecasts.drop_duplicates(subset=["date", "ticker"])
    # `pivot` is exactly right: one row per date, one column per ticker, no aggregation.
    realised = unique.pivot(index="date", columns="ticker", values="ret_1d")  # noqa: PD010
    return RunArtifacts(
        run_id=run_dir.name,
        config=cfg,
        folds=folds,
        fold_table=fold_table,
        forecasts=forecasts,
        metrics_by_fold=pd.read_csv(run_dir / "metrics_by_fold.csv"),
        metrics_summary=pd.read_csv(run_dir / "metrics_summary.csv"),
        backtests={},
        backtest_summary=pd.read_csv(run_dir / "backtest_summary.csv"),
        pnl=pd.read_parquet(run_dir / "pnl.parquet"),
        buy_and_hold=buy_and_hold(realised.sort_index(), cfg.backtest.ann_factor),
    )


def _rerender(run_dir: Path) -> Path:
    """Re-render a finished run's report from its saved artefacts.

    Args:
        run_dir: Directory holding a previous run's artefacts.

    Returns:
        The path the report module reported writing.

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist, or is missing one of
            the files :func:`_load_artifacts` needs.
        ModuleNotFoundError: If :mod:`equity_forecast.report` does not exist.
    """
    if not run_dir.is_dir():
        msg = f"no such run directory: {run_dir}"
        raise FileNotFoundError(msg)
    module = _find_report()
    if module is None:
        msg = "equity_forecast.report does not exist; nothing can be re-rendered"
        raise ModuleNotFoundError(msg)
    for name in _RERENDER_ENTRY_POINTS:
        entry = getattr(module, name, None)
        if callable(entry):
            logger.info("re-rendering %s via report.%s", run_dir, name)
            return Path(entry(run_dir))
    for name in _LOAD_ENTRY_POINTS:
        entry = getattr(module, name, None)
        if callable(entry):
            logger.info("re-rendering %s via report.%s + report.write_report", run_dir, name)
            return Path(module.write_report(entry(run_dir)))
    logger.info("rebuilding artefacts from %s and re-rendering in place", run_dir)
    return Path(module.write_report(_load_artifacts(run_dir), out_dir=run_dir))


def _quick_config() -> Path:
    """Locate ``configs/quick.yaml``.

    Returns:
        The path relative to the working directory when it exists, otherwise the
        copy that sits beside the installed source tree.
    """
    if _QUICK_CONFIG.exists():
        return _QUICK_CONFIG
    return Path(__file__).resolve().parents[2] / "configs" / "quick.yaml"


# --------------------------------------------------------------------------- #
# human-readable summary
# --------------------------------------------------------------------------- #
def _headline(artifacts: RunArtifacts) -> pd.DataFrame:
    """Build the small table printed at the end of a run.

    Args:
        artifacts: The finished run.

    Returns:
        One row per model: forecast accuracy, the Diebold-Mariano p-value
        against the random-walk baseline, and the net backtest result.
    """
    table = artifacts.metrics_summary.loc[:, list(_HEADLINE_COLUMNS)].copy()
    backtests = artifacts.backtest_summary.set_index("model")
    table["net_sharpe"] = table["model"].map(backtests["net_sharpe"])
    table["net_total_return"] = table["model"].map(backtests["net_total_return"])
    return table


def _echo_summary(artifacts: RunArtifacts, destination: Path | None) -> None:
    """Print the headline table and where the artefacts went.

    Args:
        artifacts: The finished run.
        destination: What :func:`_write_report` returned.
    """
    benchmark = summarise_pnl(
        artifacts.buy_and_hold["net"],
        turnover=artifacts.buy_and_hold["turnover"],
        ann_factor=artifacts.config.backtest.ann_factor,
    )
    typer.echo("")
    typer.echo(
        f"run {artifacts.run_id}: {len(artifacts.folds)} fold(s), "
        f"{len(artifacts.metrics_summary)} model(s)"
    )
    typer.echo(_headline(artifacts).to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    typer.echo(
        f"buy & hold benchmark: net sharpe {benchmark['sharpe']: .4f}, "
        f"total return {benchmark['total_return']: .4f}"
    )
    if destination is None:
        typer.echo("no report written (equity_forecast.report is unavailable)")
    else:
        typer.echo(f"artefacts written to {destination}")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command()
def run(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML experiment configuration.")
    ] = _DEFAULT_CONFIG,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-download prices instead of reading the cache.")
    ] = False,
    models: Annotated[
        str | None,
        typer.Option("--models", help="Comma-separated models, e.g. 'ridge,lstm'."),
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Override the run identifier.")
    ] = None,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", help="Override report.output_dir.")
    ] = None,
) -> None:
    """Run a walk-forward experiment and write its report."""
    configure_logging()
    try:
        cfg = _apply_overrides(load_config(config), run_id=run_id, results_dir=results_dir)
        artifacts = run_experiment(cfg, models=_split_models(models), refresh=refresh)
        destination = _write_report(artifacts)
    except Exception as exc:  # the CLI boundary turns anything into exit code 1
        _fail("run", exc)
    _echo_summary(artifacts, destination)


@app.command()
def report(
    run_id: Annotated[str, typer.Option("--run-id", help="Identifier of the run to re-render.")],
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Directory holding run outputs.")
    ] = Path("results"),
) -> None:
    """Re-render a previous run's report from the artefacts on disk."""
    configure_logging()
    try:
        destination = _rerender(results_dir / run_id)
    except Exception as exc:  # the CLI boundary turns anything into exit code 1
        _fail("report", exc)
    typer.echo(f"report written to {destination}")


@app.command()
def smoke(
    models: Annotated[
        str | None,
        typer.Option("--models", help="Restrict the smoke run to these models."),
    ] = None,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", help="Override report.output_dir.")
    ] = None,
) -> None:
    """Run the quick offline configuration end to end (used by CI)."""
    configure_logging()
    try:
        cfg = _apply_overrides(load_config(_quick_config()), results_dir=results_dir)
        # Belt and braces: whatever the YAML says, a smoke run never touches the network.
        cfg = replace(cfg, data=replace(cfg.data, offline=True))
        artifacts = run_experiment(cfg, models=_split_models(models), refresh=False)
        destination = _write_report(artifacts)
    except Exception as exc:  # the CLI boundary turns anything into exit code 1
        _fail("smoke", exc)
    _echo_summary(artifacts, destination)


@app.command()
def fetch(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML experiment configuration.")
    ] = _DEFAULT_CONFIG,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-download even when the cache is populated.")
    ] = False,
) -> None:
    """Populate the price cache and stop."""
    configure_logging()
    try:
        cfg = load_config(config)
        frames = download_prices(
            cfg.data.tickers,
            cfg.data.start,
            cfg.data.end,
            cfg.data.cache_dir,
            refresh,
            cfg.data.offline,
        )
    except Exception as exc:  # the CLI boundary turns anything into exit code 1
        _fail("fetch", exc)
    typer.echo(f"cached {len(frames)} ticker(s) in {cfg.data.cache_dir}")
    for ticker in sorted(frames):
        frame = frames[ticker]
        span = f"{frame.index[0].date()} -> {frame.index[-1].date()}" if len(frame) else "empty"
        typer.echo(f"  {ticker}: {len(frame)} rows, {span}")


if __name__ == "__main__":  # pragma: no cover - convenience for `python -m`
    app()
