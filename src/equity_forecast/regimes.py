"""Volatility-regime evaluation: does a model's edge depend on market conditions?

A single full-sample score can hide a model that helps in turbulent markets and
hurts in calm ones (or the reverse). This module labels every test date with a
volatility regime and re-scores each model inside each regime.

No look-ahead in the labels
---------------------------
The regime signal for the row dated ``t`` is the equal-weight cross-sectional
mean of the ``rvol_<w>`` feature - trailing realised volatility computed from
returns up to the close of ``t - 1``, i.e. exactly what a forecaster knew when
it made the forecast. The cut points that split the signal into ``low`` /
``mid`` / ``high`` are the terciles of that signal over the **training dates of
the same fold**, so a test block's labels never depend on the test block's own
volatility distribution. A consequence worth stating: a test block can be
almost entirely ``high`` (2020, 2022) or almost entirely ``low`` - the labels
describe conditions relative to the history the model was trained on, not a
balanced re-split of the test data.

Statistical caveats
-------------------
* The per-regime Diebold-Mariano test runs on the per-date cross-sectional RMS
  error, as the headline test does, but restricted to the regime's dates. Those
  dates are not contiguous, so the Newey-West correction is an approximation;
  with ``h = 1`` the loss differentials are close to serially uncorrelated and
  the approximation is mild.
* Testing every model in every regime is a family of tests. ``dm_p_holm``
  applies Holm's step-down correction across that whole family; read it, not the
  raw p-value, before calling anything significant.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from equity_forecast.metrics import (
    diebold_mariano,
    directional_accuracy,
    information_coefficient,
    rmse,
)
from equity_forecast.splits import Fold

__all__ = [
    "REGIME_COLUMNS",
    "REGIME_LABELS",
    "REGIME_METRIC_COLUMNS",
    "holm_adjust",
    "label_regimes",
    "market_vol_signal",
    "regime_metrics",
]

logger = logging.getLogger(__name__)

REGIME_LABELS: Final[tuple[str, ...]] = ("low", "mid", "high")
"""Regime names, calm to turbulent."""

REGIME_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "fold_id",
    "signal",
    "low_cut",
    "high_cut",
    "regime",
)
"""Columns of the per-date regime label frame."""

REGIME_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "regime",
    "model",
    "n_dates",
    "n_obs",
    "rmse",
    "rmse_vs_zero",
    "dir_acc",
    "ic",
    "dm_stat_vs_zero",
    "dm_p_vs_zero",
    "dm_p_holm",
    "net_sharpe",
    "net_ann_return",
)
"""Columns of :func:`regime_metrics` output."""

_QUANTILES: Final[tuple[float, float]] = (1.0 / 3.0, 2.0 / 3.0)


def market_vol_signal(dataset: pd.DataFrame, column: str) -> pd.Series:
    """Return the per-date equal-weight mean of a trailing-volatility feature.

    Args:
        dataset: The long, **unscaled** modelling dataset indexed by
            ``(date, ticker)``.
        column: The trailing realised-volatility column, e.g. ``"rvol_21"``.
            In the dataset row dated ``t`` it is built from returns up to
            ``t - 1``, so the signal is known before the forecast is made.

    Returns:
        A float series indexed by date, sorted.

    Raises:
        KeyError: If ``column`` is not in ``dataset``.
    """
    if column not in dataset.columns:
        msg = f"regime signal column {column!r} is not in the dataset"
        raise KeyError(msg)
    signal = dataset[column].groupby(level="date").mean().sort_index()
    signal.name = "signal"
    return signal.astype("float64")


def label_regimes(
    signal: pd.Series,
    folds: Sequence[Fold],
    quantiles: tuple[float, float] = _QUANTILES,
) -> pd.DataFrame:
    """Label each fold's test dates using cut points from its training dates.

    Args:
        signal: Per-date regime signal from :func:`market_vol_signal`.
        folds: The walk-forward folds.
        quantiles: Lower and upper quantile of the training-period signal that
            separate ``low`` from ``mid`` and ``mid`` from ``high``.

    Returns:
        A frame with columns :data:`REGIME_COLUMNS`, one row per test date per
        fold. A date at or below ``low_cut`` is ``low``; strictly above
        ``high_cut`` is ``high``; anything between is ``mid``.

    Raises:
        ValueError: If the quantiles are not ordered inside ``(0, 1)``.
    """
    lower, upper = quantiles
    if not 0.0 < lower < upper < 1.0:
        msg = f"regime quantiles must satisfy 0 < lower < upper < 1, got {quantiles}"
        raise ValueError(msg)
    dates = pd.DatetimeIndex(signal.index)
    frames: list[pd.DataFrame] = []
    for fold in folds:
        train = signal.loc[(dates >= fold.train_start) & (dates <= fold.train_end)].dropna()
        test = signal.loc[(dates >= fold.test_start) & (dates <= fold.test_end)].dropna()
        if train.empty or test.empty:
            logger.warning("fold %d has no regime signal to label; skipped", fold.fold_id)
            continue
        low_cut = float(train.quantile(lower))
        high_cut = float(train.quantile(upper))
        values = test.to_numpy(dtype=np.float64)
        regime = np.where(values <= low_cut, "low", np.where(values > high_cut, "high", "mid"))
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(test.index),
                    "fold_id": np.full(len(test), fold.fold_id, dtype=np.int64),
                    "signal": values,
                    "low_cut": np.full(len(test), low_cut),
                    "high_cut": np.full(len(test), high_cut),
                    "regime": regime.astype(object),
                },
                columns=list(REGIME_COLUMNS),
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(REGIME_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment of a family of p-values.

    ``NaN`` entries are ignored (they stay ``NaN``) and do not count towards the
    family size.

    Args:
        p_values: Raw p-values.

    Returns:
        Adjusted p-values, in the input order, capped at 1.
    """
    raw = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(raw.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(raw))
    m = finite.size
    if m == 0:
        return [float(v) for v in adjusted]
    order = finite[np.argsort(raw[finite], kind="stable")]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float((m - rank) * raw[index])))
        adjusted[index] = running
    return [float(v) for v in adjusted]


def _cross_sectional_rms(block: pd.DataFrame, column: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Per-date root-mean-square of ``column`` over the tickers on that date."""
    squared = block.assign(_sq=block[column].pow(2))
    rms = squared.groupby("date")["_sq"].mean().pow(0.5)
    return rms.reindex(dates)


def _sharpe(returns: pd.Series, ann_factor: int) -> tuple[float, float]:
    """Annualised Sharpe ratio and mean return of a daily return series."""
    values = returns.dropna().to_numpy(dtype=np.float64)
    if values.size < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    sharpe = mean / std * np.sqrt(ann_factor) if std > 0.0 else float("nan")
    return float(sharpe), mean * ann_factor


def regime_metrics(
    forecasts: pd.DataFrame,
    labels: pd.DataFrame,
    pnl: pd.DataFrame,
    model_names: Sequence[str],
    *,
    baseline: str,
    horizon: int,
    ann_factor: int,
    benchmark_net: pd.Series | None = None,
) -> pd.DataFrame:
    """Score every model inside every volatility regime.

    Args:
        forecasts: The long forecasts frame (``date, ticker, fold_id, model,
            forecast, ..., y_true``).
        labels: Output of :func:`label_regimes`.
        pnl: The long backtest frame (``date, model, net, ...``).
        model_names: Models to score, in output order.
        baseline: The benchmark model; errors are compared against its errors,
            or against a zero forecast when it was not run.
        horizon: Forecast horizon, for the DM Newey-West lag count.
        ann_factor: Trading days per year, for the Sharpe ratio.
        benchmark_net: Optional daily return series of the buy-and-hold
            benchmark, indexed by date; adds a ``buy_and_hold`` row per regime
            carrying only the Sharpe ratio and annualised return.

    Returns:
        A frame with columns :data:`REGIME_METRIC_COLUMNS`, one row per
        ``(regime, model)``. ``rmse_vs_zero`` is the model's pooled RMSE divided
        by the baseline's over the same rows (below 1 means lower error). The
        baseline's own DM fields are ``NaN``.
    """
    if labels.empty or forecasts.empty:
        return pd.DataFrame(columns=list(REGIME_METRIC_COLUMNS))
    keyed = labels.loc[:, ["date", "fold_id", "regime"]]
    scored = forecasts.merge(keyed, on=["date", "fold_id"], how="inner")
    base = scored.loc[scored["model"] == baseline, ["date", "ticker", "fold_id", "forecast"]]
    scored = scored.merge(
        base.rename(columns={"forecast": "baseline"}),
        on=["date", "ticker", "fold_id"],
        how="left",
    )
    scored["baseline"] = scored["baseline"].fillna(0.0)
    scored["e_model"] = scored["forecast"] - scored["y_true"]
    scored["e_base"] = scored["baseline"] - scored["y_true"]
    pnl_by_date = pnl.assign(date=pd.to_datetime(pnl["date"]))
    regime_by_date = labels.drop_duplicates("date").set_index("date")["regime"]

    rows: list[dict[str, object]] = []
    for regime in REGIME_LABELS:
        in_regime = scored.loc[scored["regime"] == regime]
        regime_dates = pd.DatetimeIndex(regime_by_date.index[regime_by_date == regime])
        for name in model_names:
            block = in_regime.loc[in_regime["model"] == name].dropna(subset=["e_model"])
            row: dict[str, object] = {
                "regime": regime,
                "model": name,
                "n_dates": int(block["date"].nunique()),
                "n_obs": len(block),
            }
            if block.empty:
                rows.append(row)
                continue
            model_rmse = rmse(block["y_true"], block["forecast"])
            base_rmse = rmse(block["y_true"], block["baseline"])
            row["rmse"] = model_rmse
            row["rmse_vs_zero"] = model_rmse / base_rmse if base_rmse > 0.0 else np.nan
            row["dir_acc"] = directional_accuracy(block["y_true"], block["forecast"])
            row["ic"] = information_coefficient(block["y_true"], block["forecast"])
            if name != baseline:
                dates = pd.DatetimeIndex(sorted(block["date"].unique()))
                e_model = _cross_sectional_rms(block, "e_model", dates)
                e_base = _cross_sectional_rms(block, "e_base", dates)
                try:
                    result = diebold_mariano(e_model.to_numpy(), e_base.to_numpy(), h=horizon)
                    row["dm_stat_vs_zero"] = result.statistic
                    row["dm_p_vs_zero"] = result.p_value
                except ValueError as exc:
                    logger.warning("regime DM skipped for %s/%s: %s", name, regime, exc)
            net = pnl_by_date.loc[pnl_by_date["model"] == name].set_index("date")["net"]
            sharpe, ann_return = _sharpe(net.reindex(regime_dates), ann_factor)
            row["net_sharpe"] = sharpe
            row["net_ann_return"] = ann_return
            rows.append(row)
        if benchmark_net is not None:
            sharpe, ann_return = _sharpe(benchmark_net.reindex(regime_dates), ann_factor)
            rows.append(
                {
                    "regime": regime,
                    "model": "buy_and_hold",
                    "n_dates": len(regime_dates),
                    "net_sharpe": sharpe,
                    "net_ann_return": ann_return,
                }
            )

    table = pd.DataFrame(rows).reindex(columns=list(REGIME_METRIC_COLUMNS))
    table["dm_p_holm"] = holm_adjust(table["dm_p_vs_zero"].astype("float64").tolist())
    return table
