"""Turn a finished experiment into artefacts a human can read and audit.

One run produces one directory, ``<output_dir>/<run_id>/``, holding the per-fold
and summary metric tables, the raw forecasts, the daily P&L, the walk-forward
fold geometry, the *resolved* configuration (so the directory is
self-describing), four figures and a ``summary.md`` that ties them together.

The narrative in ``summary.md`` is generated from the numbers, not written by
hand: :func:`generate_findings` counts folds, reads Diebold-Mariano p-values and
compares gross with net Sharpe ratios, and it is required to state a null result
plainly. A model that fails to beat the baseline is reported as failing to beat
the baseline.

This module deliberately does not import :mod:`equity_forecast.pipeline`. It
accepts anything matching :class:`RunArtifactsLike`, the structural view of
``pipeline.RunArtifacts``, which keeps the dependency pointing one way and lets
the tests build a small stand-in instead of running an experiment.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy import stats

from equity_forecast.backtest import summarise_pnl
from equity_forecast.config import load_config
from equity_forecast.metrics import dm_pvalue_matrix
from equity_forecast.splits import Fold

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from equity_forecast.backtest import BacktestResult
    from equity_forecast.config import Config

__all__ = [
    "BASELINE_LABEL",
    "BASELINE_MODEL",
    "BENCHMARK_FILE",
    "FIGURE_FILES",
    "HEADLINE_COLUMNS",
    "REPORT_FILES",
    "RunArtifactsLike",
    "build_headline_table",
    "generate_findings",
    "load_run",
    "make_plots",
    "render_summary_markdown",
    "rerender_report",
    "write_report",
]

logger = logging.getLogger(__name__)

#: Registry key of the model every challenger is measured against.
BASELINE_MODEL = "zero_return"

#: How the baseline is written in prose and column headers.
BASELINE_LABEL = "ZeroReturn"

#: Two-sided level at which a Diebold-Mariano p-value is called significant.
SIGNIFICANCE_LEVEL = 0.05

# Below this average daily turnover a book barely trades, so its return comes from
# holding a position rather than from acting on a forecast.
_STATIC_BOOK_TURNOVER = 0.05

#: Ticker label ``pipeline`` uses for rows scored across all tickers at once. Kept as a
#: literal rather than imported, so that reporting never imports the pipeline.
POOLED_TICKER = "ALL"

#: Window (trading days) of the rolling Sharpe figure; shortened on short samples.
ROLLING_SHARPE_WINDOW = 252

#: Figures written by :func:`make_plots`, in order, with their markdown captions.
FIGURE_CAPTIONS: Mapping[str, str] = {
    "cumulative_pnl.png": "Cumulative net P&L",
    "dir_acc_box.png": "Per-fold directional accuracy",
    "rolling_sharpe.png": "Rolling annualised Sharpe (net)",
    "dm_heatmap.png": "Pairwise Diebold-Mariano p-values (cross-sectional, one obs/day)",
}

#: Filenames of the figures, in the order they are drawn and embedded.
FIGURE_FILES: tuple[str, ...] = tuple(FIGURE_CAPTIONS)

#: Non-figure files :func:`write_report` always writes (FR-R1).
REPORT_FILES: tuple[str, ...] = (
    "metrics_by_fold.csv",
    "metrics_summary.csv",
    "backtest_summary.csv",
    "forecasts.parquet",
    "pnl.parquet",
    "folds.csv",
    "config.yaml",
    "summary.md",
)

#: Extra artefact written alongside :data:`REPORT_FILES`: the benchmark P&L, which nothing
#: else on disk carries and :func:`rerender_report` needs to redraw the benchmark line.
BENCHMARK_FILE = "buy_and_hold.parquet"

#: Columns of the headline table, in order.
HEADLINE_COLUMNS: tuple[str, ...] = (
    "Model",
    "RMSE (mean±std)",
    "DirAcc (mean±std)",
    "IC (mean±std)",
    f"DM p vs {BASELINE_LABEL}",
    "Net Sharpe",
    "Max DD",
    "Avg turnover",
)

_NA = "n/a"
_MIN_T_OBS = 2


class RunArtifactsLike(Protocol):
    """Structural view of ``equity_forecast.pipeline.RunArtifacts``.

    Declared as a protocol rather than imported so that :mod:`report` never
    imports :mod:`equity_forecast.pipeline` (which imports the models, the
    backtest and this module). The real dataclass satisfies it structurally.
    """

    @property
    def run_id(self) -> str:
        """Identifier of the run; also the name of the output directory."""

    @property
    def config(self) -> Config:
        """The fully resolved configuration the run was executed with."""

    @property
    def folds(self) -> list[Fold]:
        """Walk-forward folds, in chronological order."""

    @property
    def fold_table(self) -> pd.DataFrame:
        """Human-readable fold geometry, one row per fold."""

    @property
    def forecasts(self) -> pd.DataFrame:
        """Long frame: ``date, ticker, fold_id, model, forecast, ..., y_true, ret_1d``."""

    @property
    def metrics_by_fold(self) -> pd.DataFrame:
        """Long frame of accuracy metrics, one row per model/fold/ticker."""

    @property
    def metrics_summary(self) -> pd.DataFrame:
        """One row per model: ``<metric>_mean``, ``<metric>_std`` and DM columns."""

    @property
    def backtests(self) -> Mapping[str, BacktestResult]:
        """Backtest output keyed by model name."""

    @property
    def backtest_summary(self) -> pd.DataFrame:
        """One row per model of gross and net backtest statistics."""

    @property
    def pnl(self) -> pd.DataFrame:
        """Long frame: ``date, model, gross, net, cost, turnover``."""

    @property
    def buy_and_hold(self) -> pd.DataFrame:
        """Date-indexed benchmark P&L with at least a ``gross`` or ``net`` column."""


# --------------------------------------------------------------------------- #
# small frame / formatting helpers
# --------------------------------------------------------------------------- #
def _by_model(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Return ``frame`` indexed by model name, or ``None`` if it is not keyed by model."""
    if frame.empty:
        return None
    if "model" in frame.columns:
        indexed = frame.set_index("model")
    elif frame.index.name == "model":
        indexed = frame
    else:
        return None
    return indexed[~indexed.index.duplicated(keep="first")]


def _row(frame: pd.DataFrame, key: str) -> pd.Series[Any]:
    """Return one labelled row of ``frame``, collapsing any duplicate labels to the first."""
    selected = frame.loc[key]
    if isinstance(selected, pd.DataFrame):
        return selected.iloc[0]
    return selected


def _cell(row: pd.Series[Any], key: str) -> float:
    """Return ``row[key]`` as a float, or ``NaN`` if it is missing or not numeric."""
    if key not in row.index:
        return math.nan
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return math.nan


def _is_missing(value: float) -> bool:
    """Return ``True`` when ``value`` is ``NaN`` (the "metric undefined" marker)."""
    return math.isnan(value)


def _fmt_float(value: float, digits: int = 3) -> str:
    """Format a float for display, rendering ``NaN`` as ``n/a``."""
    if _is_missing(value):
        return _NA
    return f"{value:.{digits}f}"


def _fmt_mean_std(mean: float, std: float, digits: int) -> str:
    """Format a mean and its across-fold standard deviation as ``mean +/- std``."""
    if _is_missing(mean):
        return _NA
    if _is_missing(std):
        return f"{mean:.{digits}f} ± {_NA}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _cell_text(value: object) -> str:
    """Render one table cell as markdown-safe text."""
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return _NA if math.isnan(number) else f"{number:.6g}"
    if isinstance(value, int | np.integer):
        return str(int(value))
    return str(value).replace("|", "\\|")


def _rounded(frame: pd.DataFrame, places: int) -> pd.DataFrame:
    """Round every float column for display.

    The backtest table holds annualised returns, Sharpe ratios and turnovers that
    carry no meaning past the fourth decimal; printing them at six significant
    figures makes the table unreadable and implies a precision the estimates do
    not have.

    Args:
        frame: Table to round.
        places: Decimal places to keep.

    Returns:
        A copy with float columns rounded; other columns untouched.
    """
    out = frame.copy()
    float_columns = out.select_dtypes(include="number").columns
    out[float_columns] = out[float_columns].astype("float64").round(places)
    return out


def _markdown_table(frame: pd.DataFrame, *, include_index: bool | None = None) -> str:
    """Render a frame as a GitHub-flavoured markdown table.

    Args:
        frame: The table to render.
        include_index: Whether to emit the index as a leading column. ``None``
            includes it unless the index is a plain ``RangeIndex``.

    Returns:
        The table as markdown, or an italic placeholder when there are no rows.
    """
    if frame.empty:
        return "_no rows_"
    if include_index is None:
        include_index = not isinstance(frame.index, pd.RangeIndex)
    headers = [str(frame.index.name or "")] if include_index else []
    headers += [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for key, row in frame.iterrows():
        cells = [_cell_text(key)] if include_index else []
        cells += [_cell_text(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# pulling numbers out of the artefacts
# --------------------------------------------------------------------------- #
def _model_names(artifacts: RunArtifactsLike) -> list[str]:
    """Return every model name in the run, in reporting order, without duplicates."""
    names: list[str] = []
    summary = _by_model(artifacts.metrics_summary)
    if summary is not None:
        names = [str(value) for value in summary.index]
    if not names and "model" in artifacts.forecasts.columns:
        names = [str(value) for value in artifacts.forecasts["model"].drop_duplicates()]
    if not names:
        names = [str(key) for key in artifacts.backtests]
    return list(dict.fromkeys(names))


def _fold_metric_matrix(artifacts: RunArtifactsLike, column: str) -> pd.DataFrame | None:
    """Return per-fold values of ``column`` as a model-by-fold frame.

    ``metrics_by_fold`` carries one row per model, fold and ticker plus a pooled
    row scored across all tickers at once (``ticker == "ALL"``). The pooled row
    is the per-fold number, and is the one ``metrics_summary`` averages, so it is
    used whenever it is present; otherwise the fold's tickers are averaged.

    Args:
        artifacts: The run to read.
        column: Metric column name, e.g. ``"dir_acc"``.

    Returns:
        A frame indexed by model with one column per fold, or ``None`` when the
        metric was not recorded.
    """
    frame = artifacts.metrics_by_fold
    needed = {"model", "fold_id", column}
    if frame.empty or not needed <= set(frame.columns):
        return None
    if "ticker" in frame.columns:
        pooled = frame.loc[frame["ticker"] == POOLED_TICKER]
        if not pooled.empty:
            frame = pooled
    matrix = frame.pivot_table(index="model", columns="fold_id", values=column, aggfunc="mean")
    return None if matrix.empty else matrix


def _fold_values(matrix: pd.DataFrame | None, model: str) -> pd.Series[float]:
    """Return one model's per-fold values with missing folds dropped."""
    if matrix is None or model not in matrix.index:
        return pd.Series(dtype="float64")
    values = _row(matrix, model)
    return pd.Series(values.to_numpy(dtype="float64"), index=values.index).dropna()


def _metric_mean_std(artifacts: RunArtifactsLike, model: str, metric: str) -> tuple[float, float]:
    """Return the across-fold mean and standard deviation of one metric.

    Reads ``metrics_summary`` first (that is what it is for) and falls back to
    aggregating ``metrics_by_fold`` when the summary lacks the columns.

    Args:
        artifacts: The run to read.
        model: Model name.
        metric: Metric stem, e.g. ``"rmse"``.

    Returns:
        ``(mean, std)``, either element ``NaN`` when unavailable.
    """
    summary = _by_model(artifacts.metrics_summary)
    if summary is not None and model in summary.index:
        row = _row(summary, model)
        mean, std = _cell(row, f"{metric}_mean"), _cell(row, f"{metric}_std")
        if not _is_missing(mean):
            return mean, std
    values = _fold_values(_fold_metric_matrix(artifacts, metric), model)
    if values.empty:
        return math.nan, math.nan
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std


def _dm_p_value(artifacts: RunArtifactsLike, model: str) -> float:
    """Return the Diebold-Mariano p-value of ``model`` against the baseline."""
    summary = _by_model(artifacts.metrics_summary)
    if summary is None or model not in summary.index:
        return math.nan
    return _cell(_row(summary, model), "dm_p_vs_zero")


def _dm_statistic(artifacts: RunArtifactsLike, model: str) -> float:
    """Return the signed Diebold-Mariano statistic of ``model`` against the baseline.

    The sign carries the entire direction of the result: a **negative** statistic
    means the model's squared forecast errors are smaller than the baseline's,
    i.e. the model is better. A p-value on its own says only that the two error
    series differ, so nothing in this module may describe a model as beating the
    baseline without consulting this.

    Args:
        artifacts: The run to read.
        model: Model name.

    Returns:
        The signed statistic, or ``NaN`` when the run recorded no test.
    """
    summary = _by_model(artifacts.metrics_summary)
    if summary is None or model not in summary.index:
        return math.nan
    return _cell(_row(summary, model), "dm_stat_vs_zero")


def _split_by_dm_direction(
    artifacts: RunArtifactsLike, challengers: list[str]
) -> tuple[dict[str, float], list[str], list[str]]:
    """Partition challengers into significantly better and significantly worse.

    Args:
        artifacts: The run to read.
        challengers: Model names excluding the baseline.

    Returns:
        ``(p_values, better, worse)`` where ``p_values`` holds every recorded
        p-value, and the two lists hold the models that clear
        :data:`SIGNIFICANCE_LEVEL` with a negative and a positive statistic
        respectively. A model with a significant p-value but no recorded
        statistic appears in neither list: the direction is unknown, so no
        directional claim may be made about it.
    """
    p_values = {
        model: _dm_p_value(artifacts, model)
        for model in challengers
        if not _is_missing(_dm_p_value(artifacts, model))
    }
    better: list[str] = []
    worse: list[str] = []
    for model, p_value in p_values.items():
        if p_value >= SIGNIFICANCE_LEVEL:
            continue
        statistic = _dm_statistic(artifacts, model)
        if _is_missing(statistic):
            continue
        (better if statistic < 0.0 else worse).append(model)
    return p_values, better, worse


def _backtest_stat(artifacts: RunArtifactsLike, model: str, stat: str, kind: str = "net") -> float:
    """Return one backtest statistic for one model.

    Args:
        artifacts: The run to read.
        model: Model name.
        stat: A key of ``backtest.SUMMARY_KEYS``, e.g. ``"sharpe"``.
        kind: ``"net"`` (after costs) or ``"gross"``.

    Returns:
        The statistic, or ``NaN`` when the run does not carry it.
    """
    summary = _by_model(artifacts.backtest_summary)
    if summary is not None and model in summary.index:
        row = _row(summary, model)
        keys = [f"{kind}_{stat}"] + ([stat] if kind == "net" else [])
        for key in keys:
            value = _cell(row, key)
            if not _is_missing(value):
                return value
    result = artifacts.backtests.get(model)
    if result is None:
        return math.nan
    bucket = result.summary if kind == "net" else getattr(result, "gross_summary", {})
    value = bucket.get(stat, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _net_pnl_wide(artifacts: RunArtifactsLike) -> pd.DataFrame:
    """Return net daily P&L as a date-by-model frame (empty when unavailable)."""
    frame = artifacts.pnl
    if frame.empty:
        return pd.DataFrame()
    if "date" not in frame.columns:
        frame = frame.reset_index()
    if not {"date", "model", "net"} <= set(frame.columns):
        return pd.DataFrame()
    wide = frame.pivot_table(index="date", columns="model", values="net", aggfunc="mean")
    order = [name for name in _model_names(artifacts) if name in wide.columns]
    order += [str(column) for column in wide.columns if str(column) not in order]
    return wide[order].sort_index()


def _benchmark_series(artifacts: RunArtifactsLike) -> pd.Series[float] | None:
    """Return the buy-and-hold daily P&L series, preferring net over gross."""
    frame = artifacts.buy_and_hold
    if frame.empty:
        return None
    for column in ("net", "gross"):
        if column in frame.columns:
            return frame[column].astype("float64")
    return None


def _model_error_series(artifacts: RunArtifactsLike) -> dict[str, npt.NDArray[np.float64]]:
    """Return each model's per-date cross-sectional RMS forecast error.

    Errors are ``y_true - forecast`` aligned on ``(date, ticker)`` and restricted
    to rows where *every* model has a forecast, so the series are equal-length
    and directly comparable, as
    :func:`equity_forecast.metrics.dm_pvalue_matrix` requires.

    They are then collapsed to **one observation per date** by taking the
    root mean squared error across the cross-section. This matters: handing the
    DM test every ``(date, ticker)`` cell separately would treat six stocks'
    same-day errors as six independent observations when they are strongly
    cross-sectionally correlated, inflating the sample size and manufacturing
    significance - exactly the design the :mod:`equity_forecast.pipeline`
    module docstring rejects for the headline test. Squaring this series
    recovers the equal-weight cross-sectional mean squared error, so the pairwise
    heatmap and the headline ``DM p vs ZeroReturn`` column are computed on the
    same footing and cannot disagree.

    Args:
        artifacts: The run to read.

    Returns:
        Mapping of model name to a 1-D float array with one entry per date;
        empty when ``forecasts`` lacks the needed columns.
    """
    frame = artifacts.forecasts
    needed = {"model", "forecast", "y_true"}
    if frame.empty or not needed <= set(frame.columns):
        return {}
    keys = [column for column in ("date", "ticker") if column in frame.columns]
    if not keys:
        return {}
    forecast = frame.pivot_table(index=keys, columns="model", values="forecast", aggfunc="mean")
    truth = frame.pivot_table(index=keys, columns="model", values="y_true", aggfunc="mean")
    errors = (truth - forecast).dropna(how="any")
    if errors.empty:
        return {}
    if "ticker" in keys and "date" in keys:
        # Every model now covers the identical set of (date, ticker) cells, so
        # each model's daily mean is taken over the same cross-section.
        errors = errors.pow(2).groupby(level="date").mean().pow(0.5)
    order = [name for name in _model_names(artifacts) if name in errors.columns]
    return {name: np.asarray(errors[name].to_numpy(), dtype=np.float64) for name in order}


# --------------------------------------------------------------------------- #
# headline table
# --------------------------------------------------------------------------- #
def build_headline_table(artifacts: RunArtifactsLike) -> pd.DataFrame:
    """Build the one-row-per-model headline table.

    Every cell is pre-formatted text: accuracy metrics carry their across-fold
    dispersion (``mean +/- std``) so no number is ever quoted without it, and
    unavailable statistics read ``n/a`` rather than being silently dropped.

    Args:
        artifacts: The finished run.

    Returns:
        A frame with the columns of :data:`HEADLINE_COLUMNS`, one row per model,
        in reporting order.
    """
    rows: list[dict[str, str]] = []
    for model in _model_names(artifacts):
        rmse_mean, rmse_std = _metric_mean_std(artifacts, model, "rmse")
        dir_mean, dir_std = _metric_mean_std(artifacts, model, "dir_acc")
        ic_mean, ic_std = _metric_mean_std(artifacts, model, "ic")
        dm_p = math.nan if model == BASELINE_MODEL else _dm_p_value(artifacts, model)
        rows.append(
            {
                "Model": model,
                "RMSE (mean±std)": _fmt_mean_std(rmse_mean, rmse_std, 5),
                "DirAcc (mean±std)": _fmt_mean_std(dir_mean, dir_std, 4),
                "IC (mean±std)": _fmt_mean_std(ic_mean, ic_std, 4),
                f"DM p vs {BASELINE_LABEL}": _fmt_float(dm_p, 3),
                "Net Sharpe": _fmt_float(_backtest_stat(artifacts, model, "sharpe"), 2),
                "Max DD": _fmt_float(_backtest_stat(artifacts, model, "max_drawdown"), 4),
                "Avg turnover": _fmt_float(_backtest_stat(artifacts, model, "avg_turnover"), 3),
            }
        )
    return pd.DataFrame(rows, columns=list(HEADLINE_COLUMNS))


# --------------------------------------------------------------------------- #
# findings
# --------------------------------------------------------------------------- #
def _finding_fold_counts(artifacts: RunArtifactsLike, challengers: list[str]) -> str:
    """Count, per model, the folds beating the coin flip and beating the baseline."""
    dir_acc = _fold_metric_matrix(artifacts, "dir_acc")
    hit_rate = _fold_metric_matrix(artifacts, "hit_rate_vs_baseline")
    if dir_acc is None or not challengers:
        return (
            "No per-fold directional accuracy was recorded, so the fold-by-fold comparison "
            "against the coin flip could not be made."
        )
    parts: list[str] = []
    for model in challengers:
        values = _fold_values(dir_acc, model)
        if values.empty:
            continue
        wins = int((values > 0.5).sum())
        piece = f"{model} above 0.50 in {wins}/{len(values)} folds (mean {values.mean():.3f})"
        hits = _fold_values(hit_rate, model)
        if not hits.empty:
            beats = int((hits > 0.5).sum())
            piece += f", closer to the truth than {BASELINE_MODEL} in {beats}/{len(hits)} folds"
        parts.append(piece)
    if not parts:
        return "No challenger model recorded a per-fold directional accuracy."
    return "Per-fold directional accuracy: " + "; ".join(parts) + "."


def _join_names(names: list[str]) -> str:
    """Join model names into readable English (``"a"``, ``"a and b"``, ``"a, b and c"``)."""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _finding_significance(artifacts: RunArtifactsLike, challengers: list[str]) -> str:
    """State which models differ from the baseline, and in which direction.

    A p-value below 5% means the two error series differ - not that the model
    won. This sentence therefore always reports the sign of the statistic, so a
    model that is significantly *worse* than the random walk can never be
    written up as clearing a significance bar.

    Args:
        artifacts: The run to read.
        challengers: Model names excluding the baseline.

    Returns:
        One finding sentence.
    """
    p_values, better, worse = _split_by_dm_direction(artifacts, challengers)
    if not p_values:
        return (
            f"No Diebold-Mariano test against the {BASELINE_LABEL} baseline was recorded, so "
            "none of the accuracy differences in this run carry a significance statement."
        )
    listing = ", ".join(f"{model} p={value:.3f}" for model, value in p_values.items())
    best = min(p_values, key=lambda model: p_values[model])
    lead = f"Diebold-Mariano vs {BASELINE_LABEL} (squared loss): "

    clauses: list[str] = []
    if better:
        clauses.append(
            f"{_join_names(better)} beat the baseline at the {SIGNIFICANCE_LEVEL:.0%} level"
        )
    if worse:
        clauses.append(
            f"{_join_names(worse)} {'is' if len(worse) == 1 else 'are'} significantly WORSE than "
            f"the baseline - the difference is real, but it runs the wrong way"
        )
    if not clauses:
        return (
            f"{lead}no model's forecast errors are distinguishable from the baseline at the "
            f"{SIGNIFICANCE_LEVEL:.0%} level - the smallest p-value is {p_values[best]:.3f} "
            f"({best}). All p-values: {listing}."
        )
    if not better:
        clauses.append("no model beats it")
    return f"{lead}{'; '.join(clauses)}. All p-values: {listing}."


def _net_sharpes(artifacts: RunArtifactsLike, models: list[str]) -> dict[str, float]:
    """Return each model's net Sharpe ratio, skipping the ones the run did not record."""
    scores = {model: _backtest_stat(artifacts, model, "sharpe") for model in models}
    return {model: value for model, value in scores.items() if not _is_missing(value)}


def _best_traded(artifacts: RunArtifactsLike, models: list[str]) -> tuple[str, float] | None:
    """Return the highest net Sharpe among the models that actually take positions.

    The baseline is excluded when there is anything else to rank: its flat book
    scores a mechanical zero, and "the best strategy is to not trade" is a
    statement the bottom line should make in words, not by silently topping a
    leaderboard.

    Args:
        artifacts: The finished run.
        models: Every model in the run, in reporting order.

    Returns:
        ``(model, net_sharpe)`` for the best challenger, or ``None`` when no
        backtest Sharpe was recorded at all.
    """
    scores = _net_sharpes(artifacts, models)
    if not scores:
        return None
    traded = {model: value for model, value in scores.items() if model != BASELINE_MODEL} or scores
    best = max(traded, key=lambda model: traded[model])
    return best, traded[best]


def _finding_costs(artifacts: RunArtifactsLike, models: list[str]) -> str:
    """Report the best net Sharpe and quantify the turnover cost that produced it."""
    ranked = _best_traded(artifacts, models)
    if ranked is None:
        return "No backtest Sharpe ratio was recorded, so the cost drag could not be quantified."
    best, net_sharpe = ranked
    gross_sharpe = _backtest_stat(artifacts, best, "sharpe", kind="gross")
    turnover = _backtest_stat(artifacts, best, "avg_turnover")
    cost_bps = float(artifacts.config.backtest.cost_bps)
    ann_factor = int(artifacts.config.backtest.ann_factor)

    text = f"Best net Sharpe among the trading models: {best} at {net_sharpe:.2f}"
    if not _is_missing(gross_sharpe):
        text += f" (gross {gross_sharpe:.2f}, a {gross_sharpe - net_sharpe:.2f} give-up)"
    if not _is_missing(turnover):
        drag = turnover * cost_bps * ann_factor
        text += (
            f"; average daily turnover of {turnover:.2f} at {cost_bps:.1f} bps costs roughly "
            f"{drag:.0f} bps of return per year"
        )
    flat_book = _net_sharpes(artifacts, models).get(BASELINE_MODEL)
    if flat_book is not None and best != BASELINE_MODEL:
        text += f"; the do-nothing {BASELINE_MODEL} book scores {flat_book:.2f}"
    benchmark = _benchmark_series(artifacts)
    if benchmark is not None and not benchmark.empty:
        bh_sharpe = float(summarise_pnl(benchmark, ann_factor=ann_factor)["sharpe"])
        text += f"; buy-and-hold over the same window scores {bh_sharpe:.2f}"
    if net_sharpe <= 0:
        return text + ". That P&L does not survive the cost drag."
    text += ". That P&L survives the cost drag"
    if not _is_missing(turnover) and turnover < _STATIC_BOOK_TURNOVER:
        # A book that almost never trades is not reacting to a forecast, and
        # calling its return a "signal" would misdescribe what produced it.
        text += (
            f", but at {turnover:.3f} average daily turnover it is close to a static position: "
            f"it is capturing drift, not reacting to a forecast"
        )
    return text + ". Whether any accuracy edge supports it is the bottom line's question."


def _finding_ic(artifacts: RunArtifactsLike, challengers: list[str]) -> str:
    """Report mean information coefficients and whether the best is distinguishable from zero."""
    matrix = _fold_metric_matrix(artifacts, "ic")
    means = {
        model: float(_fold_values(matrix, model).mean())
        for model in challengers
        if not _fold_values(matrix, model).empty
    }
    if not means:
        return "No information coefficient was recorded, so rank correlation cannot be assessed."
    listing = ", ".join(f"{model} {value:+.4f}" for model, value in means.items())
    best = max(means, key=lambda model: means[model])
    values = _fold_values(matrix, best)
    text = f"Mean information coefficient (Spearman) per model across folds: {listing}."
    n_folds = len(values)
    std = float(values.std(ddof=1)) if n_folds > 1 else math.nan
    if n_folds >= _MIN_T_OBS and not _is_missing(std) and std > 0.0:
        t_stat = means[best] / (std / math.sqrt(n_folds))
        p_value = float(2.0 * stats.t.sf(abs(t_stat), n_folds - 1))
        verdict = (
            "distinguishable from zero"
            if p_value < SIGNIFICANCE_LEVEL
            else "not distinguishable from zero"
        )
        text += (
            f" The strongest ({best}) has a fold-level t-statistic of {t_stat:.2f} over "
            f"{n_folds} folds (p={p_value:.3f}), i.e. {verdict} at the "
            f"{SIGNIFICANCE_LEVEL:.0%} level. Read that p-value as an order of magnitude, not "
            f"as inference: walk-forward folds share most of their training data so their ICs "
            f"are not independent, and this is the best of {len(means)} models, which biases it "
            f"upward. The Diebold-Mariano tests on the daily series are the significance claims "
            f"this report stands behind."
        )
    return text


def _finding_bottom_line(
    artifacts: RunArtifactsLike, models: list[str], challengers: list[str]
) -> str:
    """Write the plain-language verdict implied by the other findings.

    Two guards keep this honest. First, "significant" means significantly
    *better* - a model whose errors differ from the baseline because it is worse
    can never contribute to a positive verdict. Second, a claimed edge must come
    from a **single model**: statistical significance from one model and a
    positive Sharpe from another are two unrelated facts, and ANDing them into
    one sentence manufactures a result that no model actually delivered.

    Args:
        artifacts: The run to read.
        models: Every model name, baseline included.
        challengers: Model names excluding the baseline.

    Returns:
        The bottom-line finding.
    """
    p_values, better, worse = _split_by_dm_direction(artifacts, challengers)
    ranked = _best_traded(artifacts, models)
    best_sharpe = math.nan if ranked is None else ranked[1]
    best_model = None if ranked is None else ranked[0]
    profitable = not _is_missing(best_sharpe) and best_sharpe > 0.0
    cost_bps = float(artifacts.config.backtest.cost_bps)
    sharpe_text = _fmt_float(best_sharpe, 2)

    # The edge must belong to one model: significantly better AND profitable.
    sharpes = _net_sharpes(artifacts, better)
    vindicated = [model for model, sharpe in sharpes.items() if sharpe > 0.0]
    worse_note = (
        f" {_join_names(worse)} {'is' if len(worse) == 1 else 'are'} significantly worse than the "
        f"baseline."
        if worse
        else ""
    )

    if ranked is None and not p_values:
        return (
            "Bottom line: this run recorded neither a significance test nor a backtest, so it "
            "supports no conclusion either way - the absence of evidence here is a gap in the "
            "run, not a result."
        )
    if vindicated:
        winner = max(vindicated, key=lambda model: sharpes[model])
        return (
            f"Bottom line: {winner} both beats the {BASELINE_LABEL} baseline on forecast accuracy "
            f"at the {SIGNIFICANCE_LEVEL:.0%} level and earns a positive net Sharpe of "
            f"{_fmt_float(sharpes[winner], 2)} after {cost_bps:.1f} bps of costs. That is a real "
            f"but small edge on one sample of one universe, and it needs out-of-sample "
            f"confirmation before it means anything.{worse_note}"
        )
    if better:
        return (
            f"Bottom line: {_join_names(better)} beat the {BASELINE_LABEL} baseline on forecast "
            f"accuracy at the {SIGNIFICANCE_LEVEL:.0%} level, but no model that does so survives "
            f"{cost_bps:.1f} bps of trading costs. A measurable accuracy edge that cannot be "
            f"traded profitably is not a strategy.{worse_note}"
        )
    if profitable:
        owner = f" ({best_model})" if best_model else ""
        return (
            f"Bottom line: the best strategy{owner} earns a net Sharpe of {sharpe_text} after "
            f"{cost_bps:.1f} bps of costs, but no model's forecast errors beat the "
            f"{BASELINE_LABEL} baseline, so that P&L is not supported by a measurable accuracy "
            f"edge - on this evidence it is drift or luck, not skill. Treat it as "
            f"unproven.{worse_note}"
        )
    return (
        f"Bottom line: nothing here beats the random walk. No model's forecast errors are "
        f"significantly better than the {BASELINE_LABEL} baseline, and the best net Sharpe after "
        f"{cost_bps:.1f} bps of costs is {sharpe_text}. On this sample the honest reading is that "
        f"daily returns are not forecastable with these features; the null result is the "
        f"result.{worse_note}"
    )


def generate_findings(artifacts: RunArtifactsLike) -> list[str]:
    """Derive the report's findings from the run's own numbers.

    Five statements are produced, in order: per-fold directional-accuracy counts,
    Diebold-Mariano significance against the baseline, the best net Sharpe with
    the turnover cost that produced it, the information coefficient and whether
    it is distinguishable from zero, and a plain-language bottom line.

    Nothing here is aspirational: when no model beats the baseline, the findings
    say exactly that.

    Args:
        artifacts: The finished run.

    Returns:
        Between three and five non-empty sentences.
    """
    models = _model_names(artifacts)
    challengers = [model for model in models if model != BASELINE_MODEL]
    findings = [
        _finding_fold_counts(artifacts, challengers),
        _finding_significance(artifacts, challengers),
        _finding_costs(artifacts, models),
        _finding_ic(artifacts, challengers),
        _finding_bottom_line(artifacts, models, challengers),
    ]
    return [text for text in findings if text]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _placeholder(ax: Axes, message: str) -> None:
    """Draw an explicit "nothing to plot" message instead of an empty frame."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_cumulative_pnl(artifacts: RunArtifactsLike, fig: Figure, ax: Axes) -> None:
    """Cumulative net P&L per model, against the buy-and-hold benchmark."""
    del fig
    net = _net_pnl_wide(artifacts)
    benchmark = _benchmark_series(artifacts)
    if net.empty and benchmark is None:
        _placeholder(ax, "no P&L recorded")
        return
    if not net.empty:
        cumulative = net.fillna(0.0).cumsum()
        for column in cumulative.columns:
            ax.plot(cumulative.index, cumulative[column], linewidth=1.4, label=str(column))
    if benchmark is not None and not benchmark.empty:
        ax.plot(
            benchmark.index,
            benchmark.fillna(0.0).cumsum(),
            linewidth=1.6,
            linestyle="--",
            color="black",
            label="buy and hold",
        )
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_title("Cumulative net P&L after costs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net log return")
    ax.legend(loc="best", fontsize=8)
    ax.grid(visible=True, alpha=0.3)


def _plot_dir_acc_box(artifacts: RunArtifactsLike, fig: Figure, ax: Axes) -> None:
    """Box plot of per-fold directional accuracy, with the 0.50 coin-flip line.

    The baseline is left out. It forecasts exactly zero, so by the convention in
    :func:`equity_forecast.metrics.directional_accuracy` it scores 0.00 in every
    fold - not a direction it got wrong, but no direction at all. Plotting that
    stretches the axis over [0, 0.6] and squeezes every model that does take a
    view into the top fifth of the figure, hiding the only thing the reader came
    to see: whether the boxes straddle 0.50.
    """
    del fig
    matrix = _fold_metric_matrix(artifacts, "dir_acc")
    data: list[Any] = []
    labels: list[str] = []
    for model in _model_names(artifacts):
        if model == BASELINE_MODEL:
            continue
        values = _fold_values(matrix, model)
        if not values.empty:
            data.append(values.to_numpy())
            labels.append(model)
    if not data:
        _placeholder(ax, "no per-fold directional accuracy recorded")
        return
    ax.boxplot(data, tick_labels=labels)
    rng = np.random.default_rng(0)
    for position, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.plot(position + jitter, values, "o", markersize=3, alpha=0.5, color="tab:blue")
    ax.axhline(0.5, color="crimson", linestyle="--", linewidth=1.2, label="coin flip (0.50)")
    ax.set_title(
        f"Directional accuracy by walk-forward fold ({BASELINE_MODEL} omitted: takes no direction)"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Directional accuracy")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", fontsize=8)
    ax.grid(visible=True, axis="y", alpha=0.3)


def _plot_rolling_sharpe(artifacts: RunArtifactsLike, fig: Figure, ax: Axes) -> None:
    """Rolling one-year annualised Sharpe ratio of each model's net P&L."""
    del fig
    net = _net_pnl_wide(artifacts)
    if net.empty:
        _placeholder(ax, "no P&L recorded")
        return
    ann_factor = int(artifacts.config.backtest.ann_factor)
    n_rows = len(net.index)
    window = ROLLING_SHARPE_WINDOW if n_rows >= ROLLING_SHARPE_WINDOW else max(2, n_rows // 2)
    if window != ROLLING_SHARPE_WINDOW:
        logger.info("rolling Sharpe window shortened to %d days (%d P&L rows)", window, n_rows)
    plotted = False
    for column in net.columns:
        series = net[column].astype("float64")
        rolling = series.rolling(window, min_periods=window)
        deviation = rolling.std(ddof=1).replace(0.0, np.nan)
        sharpe = rolling.mean() / deviation * math.sqrt(ann_factor)
        if bool(sharpe.notna().any()):
            ax.plot(sharpe.index, sharpe, linewidth=1.3, label=str(column))
            plotted = True
    if not plotted:
        _placeholder(ax, f"fewer than {window} P&L days: no rolling Sharpe")
        return
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_title(f"Rolling {window}-day annualised Sharpe (net of costs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.legend(loc="best", fontsize=8)
    ax.grid(visible=True, alpha=0.3)


def _plot_dm_heatmap(artifacts: RunArtifactsLike, fig: Figure, ax: Axes) -> None:
    """Heatmap of pairwise Diebold-Mariano p-values, annotated in each cell."""
    errors = _model_error_series(artifacts)
    if len(errors) < 2:
        _placeholder(ax, "fewer than two comparable forecast series")
        return
    horizon = int(artifacts.config.features.horizon)
    try:
        matrix = dm_pvalue_matrix(errors, h=horizon)
    except ValueError:
        logger.warning("Diebold-Mariano matrix could not be computed", exc_info=True)
        _placeholder(ax, "Diebold-Mariano p-values unavailable")
        return
    labels = [str(name) for name in matrix.index]
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="DM p-value")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = values[i, j]
            text = "-" if math.isnan(value) else f"{value:.2f}"
            colour = "white" if not math.isnan(value) and value < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=colour, fontsize=9)
    ax.set_title(f"Diebold-Mariano p-values (squared loss, h={horizon})")


def make_plots(artifacts: RunArtifactsLike, out_dir: Path) -> list[Path]:
    """Draw the four report figures into ``out_dir``.

    Uses the Agg backend so the call is safe on a headless machine, and closes
    every figure it opens. Missing inputs produce an explicit "nothing to plot"
    panel rather than an exception, so a partial run still yields a report.

    Args:
        artifacts: The finished run.
        out_dir: Directory to write the PNGs into; created if needed.

    Returns:
        The paths written, in the order of :data:`FIGURE_FILES`.
    """
    # Imported here, not at module scope: importing pyplot is expensive and selecting a
    # backend is a process-wide side effect that only a caller asking for figures should pay.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    builders = (
        _plot_cumulative_pnl,
        _plot_dir_acc_box,
        _plot_rolling_sharpe,
        _plot_dm_heatmap,
    )
    written: list[Path] = []
    for name, builder in zip(FIGURE_FILES, builders, strict=True):
        path = out_dir / name
        figure, ax = plt.subplots(figsize=(10, 6))
        try:
            builder(artifacts, figure, ax)
            figure.tight_layout()
            figure.savefig(path, dpi=110)
        finally:
            plt.close(figure)
        written.append(path)
    logger.info("wrote %d figures to %s", len(written), out_dir)
    return written


# --------------------------------------------------------------------------- #
# summary.md
# --------------------------------------------------------------------------- #
def _date_range(artifacts: RunArtifactsLike) -> str:
    """Return the evaluated date range as ``start -> end``, from the forecasts if possible."""
    frame = artifacts.forecasts
    if not frame.empty and "date" in frame.columns:
        dates = pd.to_datetime(frame["date"])
        if not dates.empty:
            return f"{dates.min():%Y-%m-%d} -> {dates.max():%Y-%m-%d}"
    data = artifacts.config.data
    return f"{data.start} -> {data.end or 'today'}"


def _tickers(artifacts: RunArtifactsLike) -> str:
    """Return the traded tickers as a comma-separated string."""
    configured = tuple(artifacts.config.data.tickers)
    if configured:
        return ", ".join(configured)
    frame = artifacts.forecasts
    if not frame.empty and "ticker" in frame.columns:
        return ", ".join(str(value) for value in frame["ticker"].drop_duplicates())
    return _NA


def _metadata_table(artifacts: RunArtifactsLike) -> pd.DataFrame:
    """Build the run-metadata table (FR-R3: id, tickers, dates, folds, costs, seed)."""
    cfg = artifacts.config
    n_folds = len(artifacts.folds) or len(artifacts.fold_table)
    vol_target = cfg.backtest.vol_target
    cost = (
        f"{cfg.backtest.cost_bps:.1f} bps per unit of turnover; "
        f"entry threshold {cfg.backtest.threshold:g}; "
        f"vol target {'none' if vol_target is None else f'{vol_target:g}'}"
    )
    rows = [
        ("Run ID", artifacts.run_id),
        ("Tickers", _tickers(artifacts)),
        ("Evaluated range", _date_range(artifacts)),
        ("Folds", str(n_folds)),
        ("Models", ", ".join(_model_names(artifacts)) or _NA),
        ("Forecast horizon", f"{cfg.features.horizon} trading day(s)"),
        (
            "Split",
            f"{cfg.splits.mode}, {cfg.splits.initial_train_years:g}y initial train, "
            f"{cfg.splits.test_months}m test blocks, {cfg.splits.gap_days}d embargo",
        ),
        ("Cost assumption", cost),
        ("Seed", str(cfg.run.seed)),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _provenance_banner(artifacts: RunArtifactsLike) -> list[str]:
    """Render the run's provenance note as a blockquote, if it has one.

    A results file that does not say where its data came from invites the reader
    to assume it came from the market. When ``report.note`` is set the note is
    the first thing in the document, above every number.

    Args:
        artifacts: The finished run.

    Returns:
        Markdown lines, empty when the config carries no note.
    """
    note = artifacts.config.report.note
    if not note:
        return []
    body = " ".join(str(note).split())
    return [f"> {body}", ""]


def render_summary_markdown(
    artifacts: RunArtifactsLike, *, figures: Sequence[str] | None = None
) -> str:
    """Render ``summary.md``: metadata, headline table, findings, tables, figures.

    Args:
        artifacts: The finished run.
        figures: Figure filenames to embed as relative image links. ``None``
            embeds all of :data:`FIGURE_FILES`; pass ``[]`` when plotting was
            disabled so the document carries no dangling links.

    Returns:
        The complete markdown document.
    """
    names = list(FIGURE_FILES if figures is None else figures)
    n_folds = len(artifacts.folds) or len(artifacts.fold_table)
    lines = [
        f"# equity-forecast run `{artifacts.run_id}`",
        "",
        *_provenance_banner(artifacts),
        "## Run metadata",
        "",
        _markdown_table(_metadata_table(artifacts), include_index=False),
        "",
        "## Headline",
        "",
        _markdown_table(build_headline_table(artifacts), include_index=False),
        "",
        f"_RMSE, DirAcc and IC are means ± standard deviations across the {n_folds} "
        f"walk-forward folds. `DM p vs {BASELINE_LABEL}` is the Diebold-Mariano p-value of the "
        f"model's squared forecast errors against the {BASELINE_MODEL} baseline; a value above "
        f"{SIGNIFICANCE_LEVEL:g} means the difference is indistinguishable from noise. Net "
        "Sharpe, max drawdown and average turnover come from the cost-aware backtest. "
        f"`{BASELINE_MODEL}` shows DirAcc 0.00 by construction, not because it is "
        "anti-predictive: it forecasts exactly zero, expressing no direction at all, and an "
        "exactly-zero forecast is scored as a miss rather than credited with a coin flip. Its "
        "IC is undefined for the same reason - a constant forecast has no rank correlation._",
        "",
        "## Findings",
        "",
    ]
    lines += [f"{i}. {text}" for i, text in enumerate(generate_findings(artifacts), start=1)]
    lines += [
        "",
        "## Walk-forward folds",
        "",
        _markdown_table(artifacts.fold_table),
        "",
        "## Backtest",
        "",
        _markdown_table(_rounded(artifacts.backtest_summary, 4)),
        "",
    ]
    if names:
        lines += ["## Figures", ""]
        for name in names:
            caption = FIGURE_CAPTIONS.get(name, name)
            lines += [f"### {caption}", "", f"![{caption}]({name})", ""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# writing everything out
# --------------------------------------------------------------------------- #
def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame to CSV, keeping the index only when it carries information."""
    frame.to_csv(path, index=not isinstance(frame.index, pd.RangeIndex))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame to Parquet with pyarrow."""
    frame.to_parquet(path, engine="pyarrow")


def _plain(value: object) -> object:
    """Convert dataclass output into YAML-safe builtins (tuples become lists)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_config(cfg: Config, path: Path) -> None:
    """Write the resolved configuration so the run directory is self-describing."""
    payload = _plain(asdict(cfg))
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)


def _copy_to_latest(run_dir: Path, latest_dir: Path) -> list[Path]:
    """Mirror the human-readable artefacts of ``run_dir`` into ``latest_dir``."""
    if latest_dir.resolve() == run_dir.resolve():
        logger.debug("latest_dir is the run directory; nothing to copy")
        return []
    latest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for pattern in ("summary.md", "*.png", "*.csv", "config.yaml"):
        for source in sorted(run_dir.glob(pattern)):
            destination = latest_dir / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    logger.info("copied %d files to %s", len(copied), latest_dir)
    return copied


def write_report(
    artifacts: RunArtifactsLike, out_dir: Path | None = None, copy_to_latest: bool = True
) -> Path:
    """Write every artefact of one run to disk (FR-R1).

    Writes ``metrics_by_fold.csv``, ``metrics_summary.csv``, ``backtest_summary.csv``,
    ``forecasts.parquet``, ``pnl.parquet``, ``folds.csv``, ``config.yaml`` and
    ``summary.md``, plus the four figures when ``report.make_plots`` is enabled
    and :data:`BENCHMARK_FILE`, which is what lets :func:`rerender_report`
    redraw the benchmark line without re-running the experiment.

    Args:
        artifacts: The finished run.
        out_dir: Directory to write into. ``None`` uses
            ``<config.report.output_dir>/<run_id>``; an explicit path is used
            verbatim, which is what the tests rely on.
        copy_to_latest: Also mirror ``summary.md``, the CSVs, ``config.yaml`` and
            the figures into ``config.report.latest_dir``.

    Returns:
        The directory everything was written to.
    """
    report_cfg = artifacts.config.report
    run_dir = Path(report_cfg.output_dir) / artifacts.run_id if out_dir is None else Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(artifacts.metrics_by_fold, run_dir / "metrics_by_fold.csv")
    _write_csv(artifacts.metrics_summary, run_dir / "metrics_summary.csv")
    _write_csv(artifacts.backtest_summary, run_dir / "backtest_summary.csv")
    _write_csv(artifacts.fold_table, run_dir / "folds.csv")
    _write_parquet(artifacts.forecasts, run_dir / "forecasts.parquet")
    _write_parquet(artifacts.pnl, run_dir / "pnl.parquet")
    _write_parquet(artifacts.buy_and_hold, run_dir / BENCHMARK_FILE)
    _write_config(artifacts.config, run_dir / "config.yaml")

    figures = make_plots(artifacts, run_dir) if report_cfg.make_plots else []
    summary = render_summary_markdown(artifacts, figures=[path.name for path in figures])
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    if copy_to_latest:
        _copy_to_latest(run_dir, Path(report_cfg.latest_dir))
    logger.info("report for run %s written to %s", artifacts.run_id, run_dir)
    return run_dir


# --------------------------------------------------------------------------- #
# reading a finished run back off disk
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _LoadedRun:
    """A run rebuilt from the files in a report directory.

    Carries everything :class:`RunArtifactsLike` promises except ``backtests``,
    whose in-memory :class:`~equity_forecast.backtest.BacktestResult` objects are
    not serialised; every statistic the report needs from them is also in
    ``backtest_summary.csv``, which is.
    """

    run_id: str
    config: Config
    folds: list[Fold]
    fold_table: pd.DataFrame
    forecasts: pd.DataFrame
    metrics_by_fold: pd.DataFrame
    metrics_summary: pd.DataFrame
    backtest_summary: pd.DataFrame
    pnl: pd.DataFrame
    buy_and_hold: pd.DataFrame
    backtests: dict[str, BacktestResult] = field(default_factory=dict)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV written by :func:`write_report`, or an empty frame if it is absent."""
    if not path.exists():
        logger.warning("%s is missing; that section of the report will be empty", path.name)
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file written by :func:`write_report`, or an empty frame if absent."""
    if not path.exists():
        logger.warning("%s is missing; that section of the report will be empty", path.name)
        return pd.DataFrame()
    return pd.read_parquet(path)


def _folds_from_table(table: pd.DataFrame) -> list[Fold]:
    """Rebuild :class:`~equity_forecast.splits.Fold` objects from ``folds.csv``."""
    columns = ("fold_id", "train_start", "train_end", "test_start", "test_end")
    if table.empty or not set(columns) <= set(table.columns):
        return []
    return [
        Fold(
            fold_id=int(row["fold_id"]),
            train_start=pd.Timestamp(row["train_start"]),
            train_end=pd.Timestamp(row["train_end"]),
            test_start=pd.Timestamp(row["test_start"]),
            test_end=pd.Timestamp(row["test_end"]),
        )
        for _, row in table.iterrows()
    ]


def load_run(run_dir: Path) -> RunArtifactsLike:
    """Read a finished run back from the directory :func:`write_report` wrote.

    Args:
        run_dir: A directory holding a previous run's artefacts.

    Returns:
        An object satisfying :class:`RunArtifactsLike`, with ``backtests`` empty
        because those objects are not serialised.

    Raises:
        FileNotFoundError: If ``run_dir`` or its ``config.yaml`` is missing.
    """
    if not run_dir.is_dir():
        msg = f"no such run directory: {run_dir}"
        raise FileNotFoundError(msg)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        msg = f"{run_dir} has no config.yaml; it was not written by write_report"
        raise FileNotFoundError(msg)
    cfg = load_config(config_path)
    fold_table = _read_csv(run_dir / "folds.csv")
    return _LoadedRun(
        run_id=cfg.run.run_id or run_dir.name,
        config=cfg,
        folds=_folds_from_table(fold_table),
        fold_table=fold_table,
        forecasts=_read_parquet(run_dir / "forecasts.parquet"),
        metrics_by_fold=_read_csv(run_dir / "metrics_by_fold.csv"),
        metrics_summary=_read_csv(run_dir / "metrics_summary.csv"),
        backtest_summary=_read_csv(run_dir / "backtest_summary.csv"),
        pnl=_read_parquet(run_dir / "pnl.parquet"),
        buy_and_hold=_read_parquet(run_dir / BENCHMARK_FILE),
    )


def rerender_report(run_dir: Path) -> Path:
    """Rebuild a run's ``summary.md`` and figures from the artefacts already on disk.

    This is what ``equity-forecast report <run_dir>`` calls: it re-reads the
    tables, re-derives the findings and redraws the figures without re-running
    the experiment, so a change to the report never requires a change to the
    numbers.

    Args:
        run_dir: A directory holding a previous run's artefacts.

    Returns:
        ``run_dir``, now holding a freshly rendered report.
    """
    logger.info("re-rendering the report in %s", run_dir)
    return write_report(load_run(run_dir), out_dir=run_dir)
