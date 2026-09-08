"""Walk-forward driver: fit, forecast, score and backtest every model.

:func:`run_experiment` is the only entry point that matters. It loads prices,
builds the feature dataset, cuts it into walk-forward folds and then, fold by
fold, fits every requested model on the training rows and asks it to predict
over the fold's whole window. Only the test-block rows are kept, so nothing a
model saw during training is ever scored.

Scaling discipline
------------------
The standardisation statistics are fitted on the **training rows of the current
fold only** and then applied to the whole window. A scaler fitted on the full
dataset would leak the test block's mean and variance into training, which
flatters every model that is sensitive to feature scale (ridge and the LSTM
especially).

Why the headline Diebold-Mariano test uses one observation per date
-------------------------------------------------------------------
The obvious way to compare a model with the ``zero_return`` baseline is to pool
every ``(date, ticker)`` squared error into a single DM test. That is wrong, and
wrong in the direction that manufactures significance: the same-day forecast
errors of six large-cap US equities are dominated by a common market factor and
are strongly cross-sectionally correlated, so ``n = n_dates * n_tickers``
massively overstates the number of *independent* observations. The standard
error shrinks by roughly ``sqrt(n_tickers)`` for no real gain in information and
p-values collapse towards zero.

The headline test therefore uses the **equal-weight cross-sectional mean of the
loss differential**: one observation per date,

    d_t = (1 / N_t) * sum_i [ L(e_model,t,i) - L(e_base,t,i) ]

which is exactly the quantity a portfolio-level user cares about, and whose
serial dependence (the only dependence left) is what Newey-West in
:func:`~equity_forecast.metrics.diebold_mariano` is designed to handle. It is
implemented by handing the DM routine each model's per-date *root mean squared*
error: under squared loss ``L(e) = e^2``, so the differential of those two
series is identically the cross-sectional mean differential above.

Per-ticker DM tests are still computed and returned in
:attr:`RunArtifacts.dm_table` for the report; they are the honest way to ask
"does this model beat the baseline on *this* stock?", where the
cross-sectional-correlation problem does not arise.

Nothing here writes to disk - :func:`equity_forecast.report.write_report` owns
that.
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from equity_forecast.backtest import (
    SUMMARY_KEYS,
    BacktestResult,
    buy_and_hold,
    run_backtest,
)
from equity_forecast.data import load_prices
from equity_forecast.features import (
    apply_scaler,
    build_dataset,
    feature_columns,
    fit_scaler,
)
from equity_forecast.metrics import (
    aggregate_metrics,
    diebold_mariano,
    directional_accuracy,
    hit_rate_vs_baseline,
    information_coefficient,
    mae,
    rmse,
)
from equity_forecast.models.base import build_model
from equity_forecast.seeding import set_global_seeds
from equity_forecast.splits import Fold, describe_folds, fold_masks, generate_folds, window_mask

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

    from equity_forecast.config import Config

__all__ = [
    "BASELINE_MODEL",
    "FORECAST_COLUMNS",
    "METRIC_COLUMNS",
    "PNL_COLUMNS",
    "POOLED_TICKER",
    "RunArtifacts",
    "resolve_models",
    "run_experiment",
]

logger = logging.getLogger(__name__)

#: Columns of the long ``forecasts`` frame, in order.
FORECAST_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "fold_id",
    "model",
    "forecast",
    "vol_forecast",
    "y_true",
    "ret_1d",
)

#: Scored metrics, in the order they appear in ``metrics_by_fold``.
METRIC_COLUMNS: tuple[str, ...] = (
    "n_obs",
    "rmse",
    "mae",
    "dir_acc",
    "ic",
    "hit_rate_vs_baseline",
)

#: The subset of :data:`METRIC_COLUMNS` that is averaged across folds.
SCORED_METRICS: tuple[str, ...] = METRIC_COLUMNS[1:]

#: Columns of the long ``pnl`` frame, in order.
PNL_COLUMNS: tuple[str, ...] = ("date", "model", "gross", "net", "cost", "turnover")

#: Pseudo-ticker used for rows pooled across the cross-section.
POOLED_TICKER: str = "ALL"

#: The random-walk benchmark every other model is measured against.
BASELINE_MODEL: str = "zero_return"

# Registry keys are populated by importing the module that defines them. Importing lazily
# keeps a ridge-only run from paying statsmodels', arch's and TensorFlow's import cost.
_MODEL_MODULES: Mapping[str, str] = {
    "zero_return": "equity_forecast.models.naive",
    "historical_mean": "equity_forecast.models.naive",
    "ridge": "equity_forecast.models.linear",
    "arima": "equity_forecast.models.arima",
    "arima_garch": "equity_forecast.models.garch",
    "lstm": "equity_forecast.models.lstm",
}

_RUN_ID_FORMAT = "%Y-%m-%dT%H-%M-%S"


@dataclass(frozen=True)
class RunArtifacts:
    """Everything one walk-forward experiment produced, in memory.

    Attributes:
        run_id: Identifier for this run; the report writes into a directory of
            this name.
        config: The configuration the run was executed with, including any
            command-line overrides.
        folds: The walk-forward folds, in chronological order.
        fold_table: :func:`~equity_forecast.splits.describe_folds` output, with
            trading-day counts taken from the dataset's own calendar.
        forecasts: Long frame with columns :data:`FORECAST_COLUMNS`; one row per
            ``(model, fold, date, ticker)`` in a fold's **test block**.
        metrics_by_fold: One row per ``(model, fold_id, ticker)`` plus one per
            ``(model, fold_id, "ALL")`` pooled across tickers, with columns
            :data:`METRIC_COLUMNS`.
        metrics_summary: One row per model: ``<metric>_mean`` / ``<metric>_std``
            over the pooled per-fold rows, ``n_folds``, and the headline
            ``dm_stat_vs_zero`` / ``dm_p_vs_zero``.
        backtests: The full :class:`~equity_forecast.backtest.BacktestResult`
            for each model, keyed by model name.
        backtest_summary: One row per model, gross and net portfolio statistics
            prefixed ``gross_`` and ``net_``.
        pnl: Long frame with columns :data:`PNL_COLUMNS`, the portfolio series
            of every model stacked.
        buy_and_hold: The equal-weight long-only benchmark, shaped like
            :attr:`BacktestResult.portfolio` (index ``date``).
        dm_table: Per-``(model, ticker)`` Diebold-Mariano results against the
            baseline, where ``ticker == "ALL"`` is the cross-sectional headline
            test described in the module docstring.
    """

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
    dm_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def resolve_models(cfg: Config, models: Sequence[str] | None) -> list[str]:
    """Resolve the list of models to run.

    Args:
        cfg: The experiment configuration.
        models: Explicit model names, or ``None`` for every model the config's
            ``models:`` section defines. Duplicates are removed, order kept.

    Returns:
        Registry names in the order they will be run.

    Raises:
        ValueError: If a requested name is not a key of ``models:``.
    """
    configured = [f.name for f in fields(cfg.models)]
    if models is None:
        return configured
    requested = list(dict.fromkeys(models))
    unknown = [name for name in requested if name not in configured]
    if unknown:
        msg = f"unknown model(s) {unknown}; config defines {configured}"
        raise ValueError(msg)
    return requested


def _register_models(names: Sequence[str]) -> None:
    """Import the modules that register the requested models.

    Args:
        names: Registry keys about to be built.

    Raises:
        ValueError: If a name has no known implementing module.
    """
    for name in names:
        module = _MODEL_MODULES.get(name)
        if module is None:
            msg = f"no module known for model {name!r}; known: {sorted(_MODEL_MODULES)}"
            raise ValueError(msg)
        importlib.import_module(module)


def _resolve_run_id(cfg: Config) -> str:
    """Return the configured run id, or a UTC timestamp.

    Args:
        cfg: The experiment configuration.

    Returns:
        ``cfg.run.run_id`` when set and non-empty, else ``%Y-%m-%dT%H-%M-%S`` in UTC.
    """
    configured = (cfg.run.run_id or "").strip()
    if configured:
        return configured
    return dt.datetime.now(dt.UTC).strftime(_RUN_ID_FORMAT)


def _dataset_calendar(dataset: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the dataset's unique trading calendar.

    Args:
        dataset: The long dataset frame.

    Returns:
        Sorted unique dates of the ``date`` index level.
    """
    return pd.DatetimeIndex(dataset.index.get_level_values("date").unique()).sort_values()


# --------------------------------------------------------------------------- #
# the walk-forward loop
# --------------------------------------------------------------------------- #
def _fold_forecasts(
    dataset: pd.DataFrame,
    columns: list[str],
    fold: Fold,
    model_names: Sequence[str],
    cfg: Config,
) -> list[pd.DataFrame]:
    """Fit and forecast every model on one fold.

    Args:
        dataset: The full long dataset.
        columns: Feature column names (model inputs only - never ``y``/``ret_1d``).
        fold: The fold to run.
        model_names: Registry keys to fit, in order.
        cfg: The experiment configuration.

    Returns:
        One long frame per model, holding that model's test-block rows.

    Raises:
        ValueError: If a model returns a series not indexed exactly like the
            window frame it was given.
    """
    index = cast("pd.MultiIndex", dataset.index)
    train_mask, test_mask = fold_masks(index, fold)
    window = window_mask(index, fold)
    train_in_window = train_mask[window]
    test_in_window = test_mask[window]

    x_all = dataset.loc[:, columns]
    scaler = fit_scaler(x_all.loc[train_mask], cfg.features)
    x_window = apply_scaler(scaler, x_all.loc[window])
    x_train = x_window.loc[train_in_window]
    y_train = dataset["y"].loc[train_mask]

    test_index = x_window.index[test_in_window]
    targets = dataset.loc[window, ["y", "ret_1d"]].to_numpy(dtype=np.float64)[test_in_window]
    n_test = int(test_in_window.sum())

    frames: list[pd.DataFrame] = []
    for name in model_names:
        started = time.perf_counter()
        model = build_model(name, getattr(cfg.models, name), cfg.run.seed)
        model.fit(x_train, y_train)
        prediction = model.predict(x_window)
        if not prediction.index.equals(x_window.index):
            msg = f"model {name!r} returned a series not indexed like its input frame"
            raise ValueError(msg)
        volatility = model.predict_vol(x_window)

        forecast = prediction.to_numpy(dtype=np.float64)[test_in_window]
        if volatility is None:
            vol = np.full(n_test, np.nan, dtype=np.float64)
        else:
            vol = volatility.to_numpy(dtype=np.float64)[test_in_window]
        n_nan = int(np.isnan(forecast).sum())
        if n_nan:
            logger.warning(
                "model %s left %d/%d test forecast(s) NaN in fold %d",
                name,
                n_nan,
                n_test,
                fold.fold_id,
            )
        logger.info(
            "  fold_id %d | %-16s %d train -> %d test rows in %.2fs",
            fold.fold_id,
            name,
            int(train_in_window.sum()),
            n_test,
            time.perf_counter() - started,
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(test_index.get_level_values("date")),
                    "ticker": pd.Index(test_index.get_level_values("ticker")).astype(str),
                    "fold_id": np.full(n_test, fold.fold_id, dtype=np.int64),
                    "model": np.full(n_test, name, dtype=object),
                    "forecast": forecast,
                    "vol_forecast": vol,
                    "y_true": targets[:, 0],
                    "ret_1d": targets[:, 1],
                },
                columns=list(FORECAST_COLUMNS),
            )
        )
    return frames


def _collect_forecasts(
    dataset: pd.DataFrame,
    folds: Sequence[Fold],
    model_names: Sequence[str],
    cfg: Config,
) -> pd.DataFrame:
    """Run every fold and stack the per-model test-block forecasts.

    Args:
        dataset: The full long dataset.
        folds: The walk-forward folds.
        model_names: Registry keys to fit, in order.
        cfg: The experiment configuration.

    Returns:
        The long ``forecasts`` frame with columns :data:`FORECAST_COLUMNS`.
    """
    columns = feature_columns(cfg.features)
    frames: list[pd.DataFrame] = []
    for position, fold in enumerate(folds, start=1):
        logger.info(
            "fold %d/%d (fold_id %d): train %s..%s | gap %d | test %s..%s",
            position,
            len(folds),
            fold.fold_id,
            fold.train_start.date(),
            fold.train_end.date(),
            cfg.splits.gap_days,
            fold.test_start.date(),
            fold.test_end.date(),
        )
        frames.extend(_fold_forecasts(dataset, columns, fold, model_names, cfg))
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts["fold_id"] = forecasts["fold_id"].astype("int64")
    return forecasts


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _with_baseline(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Attach the baseline model's forecast to every row.

    Args:
        forecasts: The long forecasts frame.

    Returns:
        A copy with an extra ``baseline`` column. When the baseline model was
        not run, the column is ``0.0`` - which is what ``zero_return`` forecasts
        by definition, so the comparison stays meaningful.
    """
    keys = ["date", "ticker", "fold_id"]
    baseline = forecasts.loc[forecasts["model"] == BASELINE_MODEL, [*keys, "forecast"]]
    if baseline.empty:
        logger.info(
            "%r was not run; using an explicit zero forecast as the baseline",
            BASELINE_MODEL,
        )
        return forecasts.assign(baseline=0.0)
    merged = forecasts.merge(
        baseline.rename(columns={"forecast": "baseline"}),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    return merged


def _score(group: pd.DataFrame) -> dict[str, float]:
    """Score one block of aligned forecasts.

    Args:
        group: Rows sharing a model (and fold, and possibly ticker), carrying
            ``y_true``, ``forecast`` and ``baseline``.

    Returns:
        A mapping keyed by :data:`METRIC_COLUMNS`.
    """
    y_true = group["y_true"].to_numpy(dtype=np.float64)
    y_pred = group["forecast"].to_numpy(dtype=np.float64)
    y_base = group["baseline"].to_numpy(dtype=np.float64)
    return {
        "n_obs": float(len(group)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "ic": information_coefficient(y_true, y_pred),
        "hit_rate_vs_baseline": hit_rate_vs_baseline(y_true, y_pred, y_base),
    }


def _metrics_by_fold(merged: pd.DataFrame, model_names: Sequence[str]) -> pd.DataFrame:
    """Score every model on every fold, per ticker and pooled.

    Args:
        merged: Forecasts with a ``baseline`` column.
        model_names: Model order for the output rows.

    Returns:
        One row per ``(model, fold_id, ticker)`` plus one per
        ``(model, fold_id, "ALL")``, with columns
        ``model, fold_id, ticker, *METRIC_COLUMNS``.
    """
    rows: list[dict[str, object]] = []
    for name in model_names:
        block = merged.loc[merged["model"] == name]
        for fold_id, fold_rows in block.groupby("fold_id", sort=True):
            for ticker, ticker_rows in fold_rows.groupby("ticker", sort=True):
                rows.append(
                    {
                        "model": name,
                        "fold_id": int(cast("int", fold_id)),
                        "ticker": str(ticker),
                        **_score(ticker_rows),
                    }
                )
            rows.append(
                {
                    "model": name,
                    "fold_id": int(cast("int", fold_id)),
                    "ticker": POOLED_TICKER,
                    **_score(fold_rows),
                }
            )
    table = pd.DataFrame(rows, columns=["model", "fold_id", "ticker", *METRIC_COLUMNS])
    table["fold_id"] = table["fold_id"].astype("int64")
    table["n_obs"] = table["n_obs"].astype("int64")
    return table


def _cross_sectional_rms(panel: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return each date's root mean squared error across the cross-section.

    Squaring this series recovers the equal-weight cross-sectional mean of the
    squared errors, which is what makes it the right input for the pooled DM
    test (see the module docstring).

    Args:
        panel: Errors as a wide ``date x ticker`` frame; ``NaN`` cells are skipped.

    Returns:
        One value per row, ``NaN`` where a date has no finite error at all.
    """
    values = panel.to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    totals = np.where(valid, np.square(values), 0.0).sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    seen = counts > 0
    out[seen] = np.sqrt(totals[seen] / counts[seen])
    return out


def _paired_cross_sectional_rms(
    model_panel: pd.DataFrame, base_panel: pd.DataFrame
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return per-date RMS errors for two models over an identical cross-section.

    Diebold-Mariano compares two loss series observation by observation, so the
    two per-date averages have to be taken over the *same* tickers. If a model
    leaves one ticker's forecasts ``NaN`` - which happens when a fit fails and a
    fallback cannot be formed - averaging each panel over "whatever is finite in
    that panel" silently compares a four-stock mean against a six-stock mean.
    The differential is then contaminated by cross-sectional composition rather
    than by forecast quality. Masking both panels to the cells finite in *both*
    keeps the comparison like for like.

    Args:
        model_panel: Challenger errors as a wide ``date x ticker`` frame.
        base_panel: Baseline errors over the same dates and tickers.

    Returns:
        A pair of per-date RMS error arrays, aligned row for row.
    """
    aligned_model, aligned_base = model_panel.align(base_panel, join="outer")
    both_finite = np.isfinite(aligned_model.to_numpy(dtype=np.float64)) & np.isfinite(
        aligned_base.to_numpy(dtype=np.float64)
    )
    mask = pd.DataFrame(both_finite, index=aligned_model.index, columns=aligned_model.columns)
    return (
        _cross_sectional_rms(aligned_model.where(mask)),
        _cross_sectional_rms(aligned_base.where(mask)),
    )


def _dm_row(
    model: str,
    ticker: str,
    e_model: npt.NDArray[np.float64],
    e_base: npt.NDArray[np.float64],
    horizon: int,
) -> dict[str, object]:
    """Run one Diebold-Mariano comparison, tolerating degenerate samples.

    Args:
        model: Model name for the output row.
        ticker: Ticker, or :data:`POOLED_TICKER` for the cross-sectional test.
        e_model: The model's forecast errors.
        e_base: The baseline's forecast errors over the same observations.
        horizon: Forecast horizon, used for the Newey-West lag count.

    Returns:
        A row for the DM table. A sample too short to test yields ``NaN``
        statistics rather than raising.
    """
    row: dict[str, object] = {"model": model, "ticker": ticker}
    try:
        result = diebold_mariano(e_model, e_base, h=horizon)
    except ValueError as exc:
        logger.warning("DM test skipped for %s/%s: %s", model, ticker, exc)
        row.update({"dm_stat_vs_zero": np.nan, "dm_p_vs_zero": np.nan, "dm_n_obs": 0})
        return row
    row.update(
        {
            "dm_stat_vs_zero": result.statistic,
            "dm_p_vs_zero": result.p_value,
            "dm_n_obs": result.n_obs,
        }
    )
    return row


def _dm_table(
    merged: pd.DataFrame,
    model_names: Sequence[str],
    horizon: int,
) -> pd.DataFrame:
    """Compare every model with the baseline, per ticker and cross-sectionally.

    Args:
        merged: Forecasts with a ``baseline`` column.
        model_names: Model order for the output rows.
        horizon: Forecast horizon in trading days.

    Returns:
        Columns ``model, ticker, dm_stat_vs_zero, dm_p_vs_zero, dm_n_obs``. Rows
        for the baseline model itself are ``NaN``: it cannot be tested against
        itself.
    """
    errors = merged.assign(
        e_model=merged["forecast"] - merged["y_true"],
        e_base=merged["baseline"] - merged["y_true"],
    )
    rows: list[dict[str, object]] = []
    for name in model_names:
        block = errors.loc[errors["model"] == name].sort_values(["ticker", "date"])
        if name == BASELINE_MODEL:
            tickers = [*sorted(block["ticker"].unique().tolist()), POOLED_TICKER]
            rows.extend(
                {
                    "model": name,
                    "ticker": str(ticker),
                    "dm_stat_vs_zero": np.nan,
                    "dm_p_vs_zero": np.nan,
                    "dm_n_obs": 0,
                }
                for ticker in tickers
            )
            continue
        for ticker, ticker_rows in block.groupby("ticker", sort=True):
            ordered = ticker_rows.sort_values("date")
            rows.append(
                _dm_row(
                    name,
                    str(ticker),
                    ordered["e_model"].to_numpy(dtype=np.float64),
                    ordered["e_base"].to_numpy(dtype=np.float64),
                    horizon,
                )
            )
        # `pivot` is exactly right: one row per date, one column per ticker, no aggregation.
        model_panel = block.pivot(index="date", columns="ticker", values="e_model")  # noqa: PD010
        base_panel = block.pivot(index="date", columns="ticker", values="e_base")  # noqa: PD010
        model_rms, base_rms = _paired_cross_sectional_rms(
            model_panel.sort_index(), base_panel.sort_index()
        )
        rows.append(_dm_row(name, POOLED_TICKER, model_rms, base_rms, horizon))
    return pd.DataFrame(
        rows,
        columns=["model", "ticker", "dm_stat_vs_zero", "dm_p_vs_zero", "dm_n_obs"],
    )


def _metrics_summary(
    by_fold: pd.DataFrame,
    dm: pd.DataFrame,
    model_names: Sequence[str],
) -> pd.DataFrame:
    """Collapse the per-fold pooled rows into one row per model.

    Args:
        by_fold: Output of :func:`_metrics_by_fold`.
        dm: Output of :func:`_dm_table`.
        model_names: Model order for the output rows.

    Returns:
        ``model``, then ``<metric>_mean``/``<metric>_std`` for every metric in
        :data:`SCORED_METRICS`, then ``n_folds``, ``dm_stat_vs_zero`` and
        ``dm_p_vs_zero``.
    """
    pooled = by_fold.loc[by_fold["ticker"] == POOLED_TICKER]
    summary = aggregate_metrics(pooled, ["model"], list(SCORED_METRICS))
    n_folds = pooled.groupby("model", sort=False)["fold_id"].nunique()
    summary["n_folds"] = summary["model"].map(n_folds).astype("int64")

    headline = dm.loc[dm["ticker"] == POOLED_TICKER].set_index("model")
    summary["dm_stat_vs_zero"] = summary["model"].map(headline["dm_stat_vs_zero"])
    summary["dm_p_vs_zero"] = summary["model"].map(headline["dm_p_vs_zero"])

    ordered = [name for name in model_names if name in set(summary["model"])]
    return summary.set_index("model").reindex(ordered).reset_index()


# --------------------------------------------------------------------------- #
# backtesting
# --------------------------------------------------------------------------- #
def _realised_panel(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return the realised one-day return panel over the tested rows.

    Args:
        forecasts: The long forecasts frame.

    Returns:
        A wide ``date x ticker`` frame of ``ret_1d``.
    """
    unique = forecasts.drop_duplicates(subset=["date", "ticker"])
    # `pivot` is exactly right: one row per date, one column per ticker, no aggregation.
    panel = unique.pivot(index="date", columns="ticker", values="ret_1d")  # noqa: PD010
    return panel.sort_index().astype("float64")


def _model_panel(block: pd.DataFrame, column: str, like: pd.DataFrame) -> pd.DataFrame:
    """Pivot one model's long output onto the realised panel's axes.

    Args:
        block: One model's rows of the forecasts frame.
        column: Value column to pivot, ``forecast`` or ``vol_forecast``.
        like: The realised panel, whose index and columns are reused.

    Returns:
        A wide ``date x ticker`` frame aligned to ``like``.
    """
    # `pivot` is exactly right: one row per date, one column per ticker, no aggregation.
    panel = block.pivot(index="date", columns="ticker", values=column)  # noqa: PD010
    return panel.reindex(index=like.index, columns=like.columns).astype("float64")


def _run_backtests(
    forecasts: pd.DataFrame,
    model_names: Sequence[str],
    cfg: Config,
) -> tuple[dict[str, BacktestResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trade every model's forecasts and summarise the result.

    Args:
        forecasts: The long forecasts frame.
        model_names: Registry keys to backtest, in order.
        cfg: The experiment configuration; ``features.horizon`` becomes the
            holding period so overlapping bets are not stacked.

    Returns:
        ``(backtests, backtest_summary, pnl, benchmark)``.
    """
    realised = _realised_panel(forecasts)
    results: dict[str, BacktestResult] = {}
    summary_rows: list[dict[str, object]] = []
    pnl_frames: list[pd.DataFrame] = []

    for name in model_names:
        block = forecasts.loc[forecasts["model"] == name]
        signal = _model_panel(block, "forecast", realised)
        vol: pd.DataFrame | None = None
        if cfg.backtest.vol_target is not None and bool(block["vol_forecast"].notna().any()):
            vol = _model_panel(block, "vol_forecast", realised)
        result = run_backtest(
            signal,
            realised,
            cfg.backtest,
            vol_forecast=vol,
            hold_days=cfg.features.horizon,
        )
        results[name] = result
        summary_rows.append(
            {
                "model": name,
                **{f"gross_{key}": result.gross_summary.get(key, np.nan) for key in SUMMARY_KEYS},
                **{f"net_{key}": result.summary.get(key, np.nan) for key in SUMMARY_KEYS},
            }
        )
        portfolio = result.portfolio.rename_axis("date").reset_index()
        portfolio.insert(1, "model", name)
        pnl_frames.append(portfolio.loc[:, list(PNL_COLUMNS)])
        logger.info(
            "backtest %-16s net sharpe %+.3f, net total return %+.4f, avg turnover %.3f",
            name,
            result.summary.get("sharpe", float("nan")),
            result.summary.get("total_return", float("nan")),
            result.summary.get("avg_turnover", float("nan")),
        )

    columns = [
        "model",
        *(f"gross_{key}" for key in SUMMARY_KEYS),
        *(f"net_{key}" for key in SUMMARY_KEYS),
    ]
    backtest_summary = pd.DataFrame(summary_rows, columns=columns)
    pnl = (
        pd.concat(pnl_frames, ignore_index=True)
        if pnl_frames
        else pd.DataFrame(columns=list(PNL_COLUMNS))
    )
    return results, backtest_summary, pnl, buy_and_hold(realised, cfg.backtest.ann_factor)


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
def run_experiment(
    cfg: Config,
    models: Sequence[str] | None = None,
    refresh: bool = False,
) -> RunArtifacts:
    """Run the whole walk-forward experiment in memory.

    Args:
        cfg: The experiment configuration.
        models: Registry keys to run, or ``None`` for every model the config's
            ``models:`` section defines.
        refresh: Re-download prices instead of reading the cache.

    Returns:
        The populated :class:`RunArtifacts`. Nothing is written to disk; pass
        the result to :func:`equity_forecast.report.write_report` for that.

    Raises:
        ValueError: If a requested model is not configured, or a model breaks
            the indexing contract of ``predict``.
    """
    set_global_seeds(cfg.run.seed)
    run_id = _resolve_run_id(cfg)
    model_names = resolve_models(cfg, models)
    _register_models(model_names)
    logger.info("run %s: models %s, seed %d", run_id, model_names, cfg.run.seed)

    prices = load_prices(cfg.data, refresh=refresh)
    dataset = build_dataset(prices, cfg.features)
    calendar = _dataset_calendar(dataset)
    logger.info(
        "dataset: %d rows, %d dates %s..%s, %d ticker(s)",
        len(dataset),
        len(calendar),
        calendar[0].date(),
        calendar[-1].date(),
        dataset.index.get_level_values("ticker").nunique(),
    )

    folds = generate_folds(calendar, cfg.splits)
    fold_table = describe_folds(folds, dates=calendar)
    logger.info("generated %d walk-forward fold(s)", len(folds))

    forecasts = _collect_forecasts(dataset, folds, model_names, cfg)
    merged = _with_baseline(forecasts)
    by_fold = _metrics_by_fold(merged, model_names)
    dm = _dm_table(merged, model_names, cfg.features.horizon)
    summary = _metrics_summary(by_fold, dm, model_names)

    backtests, backtest_summary, pnl, benchmark = _run_backtests(forecasts, model_names, cfg)

    return RunArtifacts(
        run_id=run_id,
        config=cfg,
        folds=folds,
        fold_table=fold_table,
        forecasts=forecasts,
        metrics_by_fold=by_fold,
        metrics_summary=summary,
        backtests=backtests,
        backtest_summary=backtest_summary,
        pnl=pnl,
        buy_and_hold=benchmark,
        dm_table=dm,
    )
