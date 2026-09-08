"""Turning forecasts into a traded book, and pricing that book after costs.

A forecast is only interesting if trading it survives the friction of actually
trading it. This module is the bridge between the two: it maps a panel of
point forecasts onto positions, marks those positions to the realised returns,
charges a linear transaction cost on every unit of turnover, and reports the
result both gross and net.

Conventions used throughout
---------------------------
* **Timing.** The position held on date ``t`` is a function of the forecast
  *for* ``t`` - which, by the dataset contract, was made with information up to
  the close of ``t-1``. So ``pnl_t = position_t * realised_t`` involves no
  look-ahead.
* **Units.** Returns are daily log returns, so daily pnl figures are additive
  contributions rather than multiplicative growth factors. Every aggregate here
  is therefore built on the cumulative **sum** of daily pnl (see
  :func:`summarise_pnl` for the drawdown consequence of that choice).
* **Portfolio.** The portfolio is the equal-weight mean across tickers, and
  *every* portfolio column - gross, net, cost and turnover - is the mean of the
  corresponding per-ticker column. Aggregating consistently is what keeps the
  identity ``portfolio_net == portfolio_gross - portfolio_cost`` exact rather
  than approximate.
* **Costs.** ``turnover_t = |pos_t - pos_{t-1}|`` with ``pos_{-1} = 0``, so the
  cost of establishing the very first position is charged rather than given
  away, and ``cost_t = turnover_t * cost_bps / 1e4``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from equity_forecast.config import BacktestConfig

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "PORTFOLIO_COLUMNS",
    "SUMMARY_KEYS",
    "BacktestResult",
    "build_positions",
    "buy_and_hold",
    "run_backtest",
    "summarise_pnl",
]

logger = logging.getLogger(__name__)

#: The keys :func:`summarise_pnl` returns, in order. Reporting code can rely on
#: both the membership and the ordering of this tuple.
SUMMARY_KEYS: tuple[str, ...] = (
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "avg_turnover",
    "pct_days_in_market",
    "total_return",
    "n_days",
)

#: Columns of the portfolio frame returned by :func:`run_backtest` and
#: :func:`buy_and_hold`, in order.
PORTFOLIO_COLUMNS: tuple[str, ...] = ("gross", "net", "cost", "turnover")

_BPS_PER_UNIT = 1e4


# --------------------------------------------------------------------------- #
# positions
# --------------------------------------------------------------------------- #
def _hold_rows(values: npt.NDArray[np.float64], hold_days: int) -> npt.NDArray[np.float64]:
    """Repeat every ``hold_days``-th row of ``values`` across the rows that follow.

    Args:
        values: Position array, one row per date.
        hold_days: Number of rows a refreshed signal is held for.

    Returns:
        An array of the same shape whose rows are constant within each block of
        ``hold_days`` consecutive rows.
    """
    refresh = (np.arange(values.shape[0]) // hold_days) * hold_days
    held: npt.NDArray[np.float64] = values[refresh]
    return held


def build_positions(
    forecast: pd.DataFrame,
    *,
    threshold: float = 0.0,
    vol_forecast: pd.DataFrame | None = None,
    vol_target: float | None = None,
    cap: float = 2.0,
    hold_days: int = 1,
    ann_factor: int = 252,
) -> pd.DataFrame:
    """Map a panel of forecasts onto a panel of positions.

    The rule has three stages, applied in this order:

    1. **Sign.** The base position is ``sign(forecast)``: long when the forecast
       is positive, short when negative, flat when zero or missing. Only the
       direction is traded, never the magnitude, so a model that is
       systematically over-confident cannot lever itself up.
    2. **Threshold.** Forecasts with ``abs(forecast) < threshold`` are treated as
       noise and set flat. Raising the threshold can only ever remove positions,
       so it trades time in the market against conviction.
    3. **Volatility targeting** (only when *both* ``vol_forecast`` and
       ``vol_target`` are supplied). The position is multiplied by
       ``clip(vol_target / (vol_forecast * sqrt(ann_factor)), -cap, cap)``, which
       sizes each name towards a constant annualised risk contribution instead of
       a constant notional. Where the volatility forecast is missing, zero or
       negative the scale falls back to ``1.0``: an unusable risk estimate down-
       weights nothing and, crucially, never produces an infinite position.

    Finally, when ``hold_days > 1`` the signal is **refreshed every**
    ``hold_days`` **rows and held constant in between**. This matters for
    horizon-``h`` forecasts: a forecast of the cumulative return over the next
    ``h`` days, re-entered every day, would stack ``h`` overlapping bets on the
    same move and flatter the pnl while inflating turnover. Holding for
    ``hold_days = h`` rows makes each bet correspond to exactly one forecast.
    The held position is the *fully scaled* position, so a held book incurs no
    turnover at all between refresh dates.

    Args:
        forecast: Wide frame of point forecasts, index ``date``, columns
            ``ticker``. Rows are used in the order given.
        threshold: Minimum absolute forecast required to take a position. Must
            be non-negative.
        vol_forecast: Optional wide frame of *daily* conditional volatility
            forecasts, aligned to ``forecast``.
        vol_target: Optional annualised volatility target, e.g. ``0.10``.
        cap: Maximum absolute volatility-scaling multiplier.
        hold_days: Number of rows a refreshed signal is held for. ``1`` refreshes
            every row.
        ann_factor: Trading days per year, used to annualise ``vol_forecast``.

    Returns:
        A float64 frame of positions indexed exactly like ``forecast``.

    Raises:
        ValueError: If ``threshold`` is negative, ``cap`` is negative,
            ``hold_days`` is below ``1`` or ``ann_factor`` is not positive.
    """
    if threshold < 0.0:
        msg = f"threshold must be >= 0, got {threshold}"
        raise ValueError(msg)
    if cap < 0.0:
        msg = f"cap must be >= 0, got {cap}"
        raise ValueError(msg)
    if hold_days < 1:
        msg = f"hold_days must be >= 1, got {hold_days}"
        raise ValueError(msg)
    if ann_factor <= 0:
        msg = f"ann_factor must be > 0, got {ann_factor}"
        raise ValueError(msg)
    if not forecast.index.is_monotonic_increasing:
        logger.warning(
            "build_positions received a non-monotonic index; turnover and holding "
            "are computed in the row order supplied"
        )

    raw = np.nan_to_num(forecast.to_numpy(dtype="float64", copy=True), nan=0.0)
    positions = np.sign(raw)
    positions[np.abs(raw) < threshold] = 0.0

    if vol_forecast is not None and vol_target is not None:
        positions = positions * _vol_scale(
            vol_forecast.reindex(index=forecast.index, columns=forecast.columns),
            vol_target=vol_target,
            cap=cap,
            ann_factor=ann_factor,
        )
    elif vol_target is not None or vol_forecast is not None:
        logger.debug("volatility targeting skipped: both vol_forecast and vol_target are required")

    if hold_days > 1:
        positions = _hold_rows(positions, hold_days)

    return pd.DataFrame(positions, index=forecast.index, columns=forecast.columns)


def _vol_scale(
    vol_forecast: pd.DataFrame,
    *,
    vol_target: float,
    cap: float,
    ann_factor: int,
) -> npt.NDArray[np.float64]:
    """Return the clipped volatility-targeting multiplier for each cell.

    Args:
        vol_forecast: Daily conditional volatility forecasts, already aligned.
        vol_target: Annualised volatility target.
        cap: Maximum absolute multiplier.
        ann_factor: Trading days per year.

    Returns:
        An array of multipliers, ``1.0`` wherever the volatility forecast is
        missing or non-positive.
    """
    annualised = vol_forecast.to_numpy(dtype="float64") * math.sqrt(ann_factor)
    usable = np.isfinite(annualised) & (annualised > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(usable, vol_target / np.where(usable, annualised, 1.0), 1.0)
    n_fallback = int((~usable).sum())
    if n_fallback:
        logger.warning(
            "volatility targeting fell back to unscaled positions for %d cell(s) with "
            "missing or non-positive volatility",
            n_fallback,
        )
    return np.clip(scale, -cap, cap).astype("float64")


# --------------------------------------------------------------------------- #
# summary statistics
# --------------------------------------------------------------------------- #
def summarise_pnl(
    pnl: pd.Series[float],
    turnover: pd.Series[float] | None = None,
    positions: pd.Series[float] | None = None,
    ann_factor: int = 252,
) -> dict[str, float]:
    """Summarise a daily pnl series.

    Definitions, all on the daily pnl series ``p``:

    * ``ann_return`` - ``mean(p) * ann_factor``.
    * ``ann_vol`` - ``std(p, ddof=1) * sqrt(ann_factor)``; ``0.0`` for fewer than
      two observations.
    * ``sharpe`` - ``ann_return / ann_vol``, and ``0.0`` when ``ann_vol == 0``
      (a book that never moves has no risk-adjusted return to report).
    * ``sortino`` - ``ann_return`` over the annualised downside deviation
      ``sqrt(mean(min(p, 0) ** 2)) * sqrt(ann_factor)``, i.e. penalising only
      losses, with a zero minimum acceptable return. It is ``0.0`` when there is
      no downside at all, rather than an infinity that would poison every
      downstream mean.
    * ``max_drawdown`` - the largest peak-to-trough fall of the **cumulative sum**
      of daily pnl, reported as a positive number. The sum, not a compounded
      equity curve, is the right base here because these are additive log-return
      contributions of a book that is re-sized every day rather than a
      reinvested notional. The running peak starts at ``0``, so a loss on day one
      counts as a drawdown.
    * ``avg_turnover`` - mean of ``turnover``, or ``0.0`` if not supplied.
    * ``pct_days_in_market`` - fraction of rows of ``positions`` that are
      non-zero, or ``0.0`` if not supplied.
    * ``total_return`` - ``sum(p)``.
    * ``n_days`` - number of observations.

    Args:
        pnl: Daily pnl (per ticker or portfolio-level).
        turnover: Matching daily turnover, used only for ``avg_turnover``.
        positions: Matching daily positions (or gross exposures), used only for
            ``pct_days_in_market``.
        ann_factor: Trading days per year.

    Returns:
        A dict with exactly the keys in :data:`SUMMARY_KEYS`, all floats.

    Raises:
        ValueError: If ``ann_factor`` is not positive.
    """
    if ann_factor <= 0:
        msg = f"ann_factor must be > 0, got {ann_factor}"
        raise ValueError(msg)

    values = _finite(pnl, label="pnl")
    n_days = values.size
    if n_days == 0:
        return dict.fromkeys(SUMMARY_KEYS, 0.0)

    root = math.sqrt(ann_factor)
    ann_return = float(values.mean()) * ann_factor
    ann_vol = float(values.std(ddof=1)) * root if n_days > 1 else 0.0
    downside = float(np.sqrt(np.square(np.minimum(values, 0.0)).mean())) * root

    equity = np.concatenate((np.zeros(1), np.cumsum(values)))
    max_drawdown = float(np.max(np.maximum.accumulate(equity) - equity))

    avg_turnover = 0.0
    if turnover is not None:
        turnover_values = _finite(turnover, label="turnover")
        avg_turnover = float(turnover_values.mean()) if turnover_values.size else 0.0

    pct_days_in_market = 0.0
    if positions is not None:
        position_values = _finite(positions, label="positions")
        if position_values.size:
            pct_days_in_market = float(np.mean(position_values != 0.0))

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0.0 else 0.0,
        "sortino": ann_return / downside if downside > 0.0 else 0.0,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "pct_days_in_market": pct_days_in_market,
        "total_return": float(values.sum()),
        "n_days": float(n_days),
    }


def _finite(series: pd.Series[float], *, label: str) -> npt.NDArray[np.float64]:
    """Return ``series`` as a float array with non-finite entries replaced by zero.

    Args:
        series: The series to coerce.
        label: Name used in the warning emitted when values had to be replaced.

    Returns:
        A float64 array of the same length as ``series``.
    """
    values = np.asarray(series.to_numpy(dtype="float64"), dtype="float64")
    bad = ~np.isfinite(values)
    if bad.any():
        logger.warning("%d non-finite value(s) in %s treated as 0.0", int(bad.sum()), label)
        values = np.where(bad, 0.0, values)
    return values


# --------------------------------------------------------------------------- #
# the backtest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestResult:
    """Everything one backtest produced, per ticker and at portfolio level.

    Attributes:
        positions: Positions actually held, date x ticker.
        gross_pnl: Per-ticker daily pnl before costs.
        net_pnl: Per-ticker daily pnl after costs.
        portfolio: Equal-weight portfolio series with columns
            :data:`PORTFOLIO_COLUMNS`; ``net == gross - cost`` elementwise.
        summary: Portfolio-level statistics of the **net** pnl, keyed by
            :data:`SUMMARY_KEYS`.
        per_ticker: The same statistics computed on each ticker's net pnl, one
            row per ticker, index named ``ticker``.
        gross_summary: Portfolio-level statistics of the **gross** pnl, so a
            result can always be reported both before and after costs.
    """

    positions: pd.DataFrame
    gross_pnl: pd.DataFrame
    net_pnl: pd.DataFrame
    portfolio: pd.DataFrame
    summary: dict[str, float]
    per_ticker: pd.DataFrame
    gross_summary: dict[str, float] = field(default_factory=dict)


def run_backtest(
    forecast: pd.DataFrame,
    realised: pd.DataFrame,
    cfg: BacktestConfig,
    vol_forecast: pd.DataFrame | None = None,
    hold_days: int = 1,
) -> BacktestResult:
    """Trade a forecast panel against realised returns and price the result.

    The two panels are first aligned on the dates and tickers they share, so a
    model that only covers part of the universe is scored on the part it covers.
    Positions come from :func:`build_positions` using ``cfg``; then

    * ``gross_t = pos_t * realised_t``
    * ``turnover_t = |pos_t - pos_{t-1}|`` with ``pos_{-1} = 0``
    * ``cost_t = turnover_t * cfg.cost_bps / 1e4``
    * ``net_t = gross_t - cost_t``

    Args:
        forecast: Wide frame of point forecasts, index ``date``, columns ``ticker``.
        realised: Wide frame of realised returns over the same panel.
        cfg: Cost, threshold and volatility-target settings.
        vol_forecast: Optional wide frame of daily volatility forecasts; used only
            when ``cfg.vol_target`` is set.
        hold_days: Number of rows a signal is held for; pass the forecast horizon
            to avoid stacking overlapping bets.

    Returns:
        A populated :class:`BacktestResult`.

    Raises:
        ValueError: If the two panels share no dates or no tickers.
    """
    dates = forecast.index.intersection(realised.index).sort_values()
    tickers = forecast.columns.intersection(realised.columns).sort_values()
    if len(dates) == 0 or len(tickers) == 0:
        msg = (
            "forecast and realised share no common (date, ticker): "
            f"{len(dates)} common date(s), {len(tickers)} common ticker(s)"
        )
        raise ValueError(msg)
    if len(dates) < len(forecast.index) or len(tickers) < len(forecast.columns):
        logger.info(
            "backtesting on the %d x %d overlap of a %d x %d forecast panel",
            len(dates),
            len(tickers),
            len(forecast.index),
            len(forecast.columns),
        )

    aligned_forecast = forecast.loc[dates, tickers].astype("float64")
    aligned_realised = realised.loc[dates, tickers].astype("float64")
    if bool(aligned_realised.isna().to_numpy().any()):
        logger.warning("realised returns contain NaN; those days are traded as zero pnl")
        aligned_realised = aligned_realised.fillna(0.0)
    aligned_vol = (
        None if vol_forecast is None else vol_forecast.reindex(index=dates, columns=tickers)
    )

    positions = build_positions(
        aligned_forecast,
        threshold=cfg.threshold,
        vol_forecast=aligned_vol,
        vol_target=cfg.vol_target,
        cap=cfg.vol_scale_cap,
        hold_days=hold_days,
        ann_factor=cfg.ann_factor,
    )

    gross_pnl = positions * aligned_realised
    turnover = positions.sub(positions.shift(1, fill_value=0.0)).abs()
    cost = turnover * (cfg.cost_bps / _BPS_PER_UNIT)
    net_pnl = gross_pnl - cost

    portfolio = pd.DataFrame(
        {
            "gross": gross_pnl.mean(axis=1),
            "net": net_pnl.mean(axis=1),
            "cost": cost.mean(axis=1),
            "turnover": turnover.mean(axis=1),
        },
        columns=list(PORTFOLIO_COLUMNS),
    )
    # Gross exposure: non-zero on any day the book holds something, whatever the net side.
    exposure = positions.abs().mean(axis=1)

    per_ticker = pd.DataFrame.from_dict(
        {
            str(ticker): summarise_pnl(
                net_pnl[ticker],
                turnover=turnover[ticker],
                positions=positions[ticker],
                ann_factor=cfg.ann_factor,
            )
            for ticker in net_pnl.columns
        },
        orient="index",
    )
    per_ticker.index.name = "ticker"

    return BacktestResult(
        positions=positions,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        portfolio=portfolio,
        summary=summarise_pnl(
            portfolio["net"],
            turnover=portfolio["turnover"],
            positions=exposure,
            ann_factor=cfg.ann_factor,
        ),
        per_ticker=per_ticker,
        gross_summary=summarise_pnl(
            portfolio["gross"],
            turnover=portfolio["turnover"],
            positions=exposure,
            ann_factor=cfg.ann_factor,
        ),
    )


def buy_and_hold(realised: pd.DataFrame, ann_factor: int = 252) -> pd.DataFrame:
    """Return the equal-weight, long-only benchmark as a portfolio frame.

    Every ticker is held at a unit position for the whole sample, so the
    benchmark's daily pnl is the cross-sectional mean of realised returns. It is
    bought once - turnover is ``1`` on the first day and ``0`` thereafter - and
    reported without transaction costs, so ``net == gross``. The point of the
    series is to answer "did the model beat simply owning the universe?" on the
    same axes as :func:`run_backtest`.

    Args:
        realised: Wide frame of realised returns, index ``date``, columns
            ``ticker``.
        ann_factor: Trading days per year. Unused here, since the frame holds
            daily pnl only; accepted so callers can pass the same value they give
            :func:`summarise_pnl` without special-casing this function.

    Returns:
        A frame indexed by date with columns :data:`PORTFOLIO_COLUMNS`.
    """
    del ann_factor
    values = realised.astype("float64").sort_index().fillna(0.0)
    gross = values.mean(axis=1) if len(values.columns) else pd.Series(0.0, index=values.index)
    turnover = pd.Series(0.0, index=values.index, name="turnover")
    if len(turnover):
        turnover.iloc[0] = 1.0
    return pd.DataFrame(
        {
            "gross": gross,
            "net": gross,
            "cost": pd.Series(0.0, index=values.index),
            "turnover": turnover,
        },
        columns=list(PORTFOLIO_COLUMNS),
    )
