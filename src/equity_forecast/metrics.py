"""Forecast-accuracy metrics and formal model-comparison tests.

Two families of function live here.

*Point metrics* (:func:`rmse`, :func:`mae`, :func:`directional_accuracy`,
:func:`information_coefficient`, :func:`hit_rate_vs_baseline`) score one forecast
series against the realised target. They all accept anything array-like, coerce
to ``float64``, drop rows where *any* input is ``NaN``, and return ``NaN`` -
never a fabricated zero - when the metric is undefined for the surviving rows.

*Comparison tests* (:func:`diebold_mariano`, :func:`dm_pvalue_matrix`) ask
whether the gap between two models' errors is larger than sampling noise. A
lower RMSE is not evidence on its own; the Diebold-Mariano test is what turns it
into a claim with a p-value attached.

:func:`aggregate_metrics` collapses per-fold scores into mean and standard
deviation columns, so no headline number is ever reported without its dispersion
across walk-forward folds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "DMResult",
    "aggregate_metrics",
    "diebold_mariano",
    "directional_accuracy",
    "dm_pvalue_matrix",
    "hit_rate_vs_baseline",
    "information_coefficient",
    "mae",
    "rmse",
]

logger = logging.getLogger(__name__)

#: Anything the metrics accept: a float array, a pandas Series, or a plain sequence.
ArrayLike1D: TypeAlias = "npt.NDArray[np.float64] | pd.Series[float] | Sequence[float]"

#: Loss functions :func:`diebold_mariano` understands.
LOSS_FUNCTIONS: frozenset[str] = frozenset({"squared", "abs"})

_MIN_IC_OBS = 3
_MIN_DM_OBS = 2


# --------------------------------------------------------------------------- #
# coercion helpers
# --------------------------------------------------------------------------- #
def _as_float_array(values: ArrayLike1D, name: str) -> npt.NDArray[np.float64]:
    """Coerce one input to a 1-D ``float64`` array.

    Args:
        values: Array, Series or sequence of numbers.
        name: Argument name, used in the error message.

    Returns:
        A 1-D ``float64`` array (a view when the input already was one).

    Raises:
        ValueError: If the coerced array is not one-dimensional.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        msg = f"{name} must be one-dimensional, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def _align(*named: tuple[str, ArrayLike1D]) -> tuple[npt.NDArray[np.float64], ...]:
    """Coerce several inputs to equal-length float arrays and drop NaN rows pairwise.

    A row is kept only if every input is finite-or-infinite but not ``NaN`` at
    that position, so all returned arrays stay index-aligned.

    Args:
        *named: ``(name, values)`` pairs, in the order the caller declared them.

    Returns:
        The coerced arrays, in input order, restricted to the rows no input left ``NaN``.

    Raises:
        ValueError: If the inputs do not all have the same length, or any is not 1-D.
    """
    arrays = [_as_float_array(values, name) for name, values in named]
    if len({arr.size for arr in arrays}) > 1:
        sizes = ", ".join(
            f"{name}={arr.size}" for (name, _), arr in zip(named, arrays, strict=True)
        )
        msg = f"inputs must have the same length, got {sizes}"
        raise ValueError(msg)
    keep = np.ones(arrays[0].size, dtype=bool)
    for arr in arrays:
        keep &= ~np.isnan(arr)
    return tuple(arr[keep] for arr in arrays)


# --------------------------------------------------------------------------- #
# point metrics
# --------------------------------------------------------------------------- #
def rmse(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    """Return the root mean squared error.

    Args:
        y_true: Realised target values.
        y_pred: Forecast values, aligned with ``y_true``.

    Returns:
        ``sqrt(mean((y_true - y_pred) ** 2))`` over rows where neither input is
        ``NaN``, or ``NaN`` if no such row exists.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    truth, pred = _align(("y_true", y_true), ("y_pred", y_pred))
    if truth.size == 0:
        return math.nan
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mae(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    """Return the mean absolute error.

    Args:
        y_true: Realised target values.
        y_pred: Forecast values, aligned with ``y_true``.

    Returns:
        ``mean(|y_true - y_pred|)`` over rows where neither input is ``NaN``, or
        ``NaN`` if no such row exists.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    truth, pred = _align(("y_true", y_true), ("y_pred", y_pred))
    if truth.size == 0:
        return math.nan
    return float(np.mean(np.abs(truth - pred)))


def directional_accuracy(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    """Return the fraction of observations whose direction the forecast got right.

    Rows with ``y_true == 0`` are excluded: there is no direction to get right,
    and counting them would reward or punish a model for nothing.

    A forecast of **exactly zero counts as a miss**, because ``sign(0) == 0``
    never equals ``sign(y_true) in {-1, +1}``. This is deliberate. The
    ``zero_return`` baseline predicts exactly ``0.0`` everywhere, and any
    convention that credited an unsigned forecast - say, half a point for a tie -
    would hand that baseline a spurious ~50% directional accuracy and make it
    look competitive with models that actually take a side.

    Args:
        y_true: Realised target values.
        y_pred: Forecast values, aligned with ``y_true``.

    Returns:
        The fraction in ``[0, 1]`` of non-zero-target rows where
        ``sign(y_pred) == sign(y_true)``, or ``NaN`` if no row qualifies.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    truth, pred = _align(("y_true", y_true), ("y_pred", y_pred))
    directional = truth != 0.0
    if not directional.any():
        return math.nan
    hits = np.sign(pred[directional]) == np.sign(truth[directional])
    return float(np.mean(hits))


def information_coefficient(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    """Return the information coefficient: Spearman rank correlation of forecast and outcome.

    Rank correlation is used rather than Pearson because the quantity that
    matters for a cross-sectional trading rule is whether the forecast *orders*
    outcomes correctly, and because daily returns are heavy-tailed enough that a
    handful of observations would otherwise dominate a linear correlation.

    Args:
        y_true: Realised target values.
        y_pred: Forecast values, aligned with ``y_true``.

    Returns:
        Spearman rho in ``[-1, 1]``, or ``NaN`` when the statistic is undefined:
        fewer than three usable rows, or either input constant.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    truth, pred = _align(("y_true", y_true), ("y_pred", y_pred))
    if truth.size < _MIN_IC_OBS:
        return math.nan
    # scipy warns and returns NaN on constant input; check first so callers get a
    # clean NaN instead of a warning they have to filter.
    if bool(np.all(truth == truth[0])) or bool(np.all(pred == pred[0])):
        return math.nan
    return float(stats.spearmanr(truth, pred).statistic)


def hit_rate_vs_baseline(
    y_true: ArrayLike1D,
    y_pred: ArrayLike1D,
    y_base: ArrayLike1D,
) -> float:
    """Return the fraction of rows on which the model beats a baseline forecast.

    "Beats" means a strictly smaller absolute error, so ties (including two
    identical forecasts) count against the model.

    Args:
        y_true: Realised target values.
        y_pred: Model forecast, aligned with ``y_true``.
        y_base: Baseline forecast, aligned with ``y_true``.

    Returns:
        The fraction in ``[0, 1]`` of rows where
        ``|y_pred - y_true| < |y_base - y_true|``, or ``NaN`` if no row survives
        NaN-dropping.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    truth, pred, base = _align(("y_true", y_true), ("y_pred", y_pred), ("y_base", y_base))
    if truth.size == 0:
        return math.nan
    return float(np.mean(np.abs(pred - truth) < np.abs(base - truth)))


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DMResult:
    """Outcome of a Diebold-Mariano test.

    Attributes:
        statistic: The test statistic. **Negative means model 1 has the lower
            loss**, i.e. ``e1`` is the better error series.
        p_value: Two-sided p-value from the t distribution with ``n_obs - 1``
            degrees of freedom.
        n_obs: Number of paired observations actually used (after NaN-dropping).
        harvey_corrected: Whether the Harvey-Leybourne-Newbold small-sample
            factor was applied.
    """

    statistic: float
    p_value: float
    n_obs: int
    harvey_corrected: bool


def _apply_loss(errors: npt.NDArray[np.float64], loss: str) -> npt.NDArray[np.float64]:
    """Apply the requested loss function elementwise.

    Args:
        errors: Forecast errors.
        loss: ``"squared"`` or ``"abs"``.

    Returns:
        The elementwise loss.

    Raises:
        ValueError: If ``loss`` is not a known loss function.
    """
    if loss == "squared":
        return np.square(errors)
    if loss == "abs":
        return np.abs(errors)
    msg = f"unknown loss {loss!r}; expected one of {sorted(LOSS_FUNCTIONS)}"
    raise ValueError(msg)


def _autocovariance(deviations: npt.NDArray[np.float64], lag: int) -> float:
    """Return the sample autocovariance of a demeaned series at ``lag``.

    Uses the ``1/n`` (biased) divisor at every lag, as in the original
    Diebold-Mariano paper.

    Args:
        deviations: The series minus its own mean.
        lag: Non-negative lag.

    Returns:
        ``gamma_lag``; ``0.0`` when the lag is at least the series length.
    """
    n = deviations.size
    if lag == 0:
        return float(np.mean(deviations * deviations))
    if lag >= n:
        return 0.0
    return float(np.sum(deviations[lag:] * deviations[:-lag]) / n)


def diebold_mariano(
    e1: ArrayLike1D,
    e2: ArrayLike1D,
    h: int = 1,
    loss: str = "squared",
    harvey_correction: bool = True,
) -> DMResult:
    """Test whether two forecast error series have significantly different expected loss.

    The loss differential is ``d_t = L(e1_t) - L(e2_t)``. Its long-run variance is
    estimated by truncating the autocovariance sum at ``h - 1`` lags,
    ``lrv = gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k`` (for the usual ``h = 1`` this
    is just ``gamma_0``), giving ``DM = mean(d) / sqrt(lrv / n)``. With
    ``harvey_correction`` the statistic is multiplied by the
    Harvey-Leybourne-Newbold small-sample factor
    ``sqrt((n + 1 - 2h + h(h - 1) / n) / n)``, which is below 1 and therefore
    makes the test *more* conservative. The p-value is two-sided from the t
    distribution with ``n - 1`` degrees of freedom.

    **Sign convention: a negative statistic means model 1 is better** (``e1``
    carries the lower loss).

    Degenerate cases are reported as "no evidence" rather than as an error or an
    infinity: if ``d`` has zero variance - most commonly because the two models
    produced identical errors - the statistic is undefined, and this returns
    ``statistic=0.0, p_value=1.0``.

    Args:
        e1: Forecast errors of model 1 (``y_pred - y_true``, or the reverse; the
            loss functions are symmetric in sign).
        e2: Forecast errors of model 2, aligned with ``e1``.
        h: Forecast horizon in periods; ``h - 1`` autocovariance lags enter the
            long-run variance. Must be at least 1.
        loss: ``"squared"`` or ``"abs"``.
        harvey_correction: Whether to apply the Harvey-Leybourne-Newbold factor.

    Returns:
        A :class:`DMResult`.

    Raises:
        ValueError: If ``h < 1``, ``loss`` is unknown, the two series have
            different lengths, or fewer than two paired observations survive
            NaN-dropping.
    """
    if h < 1:
        msg = f"h must be >= 1, got {h}"
        raise ValueError(msg)
    if loss not in LOSS_FUNCTIONS:
        msg = f"unknown loss {loss!r}; expected one of {sorted(LOSS_FUNCTIONS)}"
        raise ValueError(msg)

    err1, err2 = _align(("e1", e1), ("e2", e2))
    n = int(err1.size)
    if n < _MIN_DM_OBS:
        msg = f"Diebold-Mariano needs at least {_MIN_DM_OBS} paired observations, got {n}"
        raise ValueError(msg)

    d = _apply_loss(err1, loss) - _apply_loss(err2, loss)
    dbar = float(np.mean(d))
    deviations = d - dbar

    gamma_0 = _autocovariance(deviations, 0)
    if gamma_0 <= 0.0:
        logger.debug("loss differential has zero variance (n=%d); DM statistic is undefined", n)
        return DMResult(statistic=0.0, p_value=1.0, n_obs=n, harvey_corrected=False)

    lrv = gamma_0 + 2.0 * sum(_autocovariance(deviations, k) for k in range(1, h))
    if lrv <= 0.0:
        logger.warning(
            "long-run variance is non-positive (%.6g) at h=%d; falling back to gamma_0", lrv, h
        )
        lrv = gamma_0

    statistic = dbar / math.sqrt(lrv / n)

    corrected = False
    if harvey_correction:
        factor_squared = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if factor_squared > 0.0:
            statistic *= math.sqrt(factor_squared)
            corrected = True
        else:
            logger.warning(
                "Harvey correction undefined for n=%d, h=%d; reporting the uncorrected statistic",
                n,
                h,
            )

    p_value = float(2.0 * stats.t.sf(abs(statistic), n - 1))
    return DMResult(
        statistic=float(statistic),
        p_value=p_value,
        n_obs=n,
        harvey_corrected=corrected,
    )


def dm_pvalue_matrix(
    errors: Mapping[str, ArrayLike1D],
    h: int = 1,
    loss: str = "squared",
) -> pd.DataFrame:
    """Build the symmetric matrix of pairwise Diebold-Mariano p-values.

    Intended for the report's significance heatmap: cell ``(i, j)`` is the
    probability of seeing a loss gap at least as large as the observed one
    between models ``i`` and ``j`` if their expected losses were equal. The
    p-value is symmetric because it depends only on ``|statistic|``, and the
    diagonal is ``NaN`` because a model cannot be compared with itself.

    Args:
        errors: Mapping of model name to that model's error series. All series
            must have the same length; iteration order sets the row/column order.
        h: Forecast horizon passed to :func:`diebold_mariano`.
        loss: Loss function passed to :func:`diebold_mariano`.

    Returns:
        A square ``float64`` frame indexed and columned by model name, symmetric,
        with a ``NaN`` diagonal.

    Raises:
        ValueError: Propagated from :func:`diebold_mariano` for any invalid pair.
    """
    names = list(errors)
    axis = pd.Index(names, name="model")
    matrix = pd.DataFrame(np.nan, index=axis, columns=axis, dtype="float64")
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            p_value = diebold_mariano(errors[left], errors[right], h=h, loss=loss).p_value
            matrix.loc[left, right] = p_value
            matrix.loc[right, left] = p_value
    return matrix


# --------------------------------------------------------------------------- #
# aggregation across folds
# --------------------------------------------------------------------------- #
def aggregate_metrics(
    by_fold: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Summarise per-fold metrics as mean and standard deviation.

    A walk-forward mean on its own hides how unstable a model is across folds, so
    every metric comes back paired with its dispersion: ``<metric>_mean`` and
    ``<metric>_std`` (sample standard deviation, ``ddof=1``, hence ``NaN`` for a
    single fold - which is the honest answer).

    Args:
        by_fold: Long frame of per-fold scores, one row per (group, fold).
        group_cols: Columns identifying a group, e.g. ``["model"]`` or
            ``["model", "ticker"]``. Groups are returned as ordinary columns.
        metric_cols: Numeric columns to summarise.

    Returns:
        One row per group, with ``group_cols`` first and then ``<metric>_mean``,
        ``<metric>_std`` for each metric in ``metric_cols`` order.

    Raises:
        ValueError: If ``group_cols`` or ``metric_cols`` is empty.
        KeyError: If any named column is missing from ``by_fold``.
    """
    if not group_cols:
        msg = "group_cols must name at least one column"
        raise ValueError(msg)
    if not metric_cols:
        msg = "metric_cols must name at least one column"
        raise ValueError(msg)
    missing = [c for c in (*group_cols, *metric_cols) if c not in by_fold.columns]
    if missing:
        msg = f"columns missing from by_fold: {missing}"
        raise KeyError(msg)

    grouped = by_fold.groupby(group_cols, dropna=False, observed=True)[metric_cols].agg(
        ["mean", "std"]
    )
    # `agg` lays the result out metric-major, so this reproduces its column order exactly.
    grouped.columns = pd.Index(
        [f"{metric}_{suffix}" for metric in metric_cols for suffix in ("mean", "std")]
    )
    return grouped.reset_index()
