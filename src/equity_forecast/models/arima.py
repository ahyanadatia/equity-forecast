"""ARIMA on returns, with fixed parameters and genuine one-step-ahead forecasts.

Why this module is written the way it is
----------------------------------------
The naive way to use ARIMA in a walk-forward study is to fit on the training
block and then call ``get_forecast(len(test))``. That is a **multi-step**
forecast, and for a stationary ARMA it decays geometrically to the unconditional
mean within a handful of days: an ARMA(1, 0, 1) with an AR root of 0.3 is 99%
of the way back to its mean after four steps. The resulting "forecast" is a flat
line, which flatters the error metrics (it is barely distinguishable from the
historical-mean baseline) while forecasting nothing at all. Every day of the
test block would be predicted with information that is by then weeks stale.

The honest alternative is a rolling one-step-ahead forecast: on each day ``t``,
condition on returns up to ``t-1`` and predict ``t``. Doing that by re-estimating
the model every single day is the textbook approach, and it is also the reason
so many ARIMA backtests take hours - a six-month test block means ~126 maximum
likelihood fits per ticker per fold.

**The design decision taken here: parameters are estimated ONCE PER FOLD, on the
training block, and then held fixed while the model is re-filtered over the
prediction window to produce true one-step-ahead forecasts.** Concretely,
:meth:`ArimaForecaster.predict` recovers the realised return series over the
window with :func:`equity_forecast.models.base.returns_from_features`, splices it
onto the training returns, and calls statsmodels' ``results.apply(new_endog,
refit=False)``. ``apply`` re-runs the Kalman filter over the new data with the
*old* parameter vector; the in-sample predictions it produces
(``get_prediction(...).predicted_mean``, equivalently ``fittedvalues``) are
one-step-ahead predictions - the value at ``t`` is ``E[r_t | r_{t-1}, r_{t-2},
...]``, which uses no observation from ``t`` onwards.

*The trade-off.* Fixing the parameters within a fold means the coefficients are
up to one test block stale by the end of that block (six months, with the
default split geometry). Daily re-estimation would let them drift with the data.
In exchange we get a run that is ~100x cheaper, a model whose behaviour inside a
fold is perfectly reproducible, and - most importantly - an apples-to-apples
comparison with every other model in the package, all of which are also fitted
once per fold and then applied. Given that ARMA coefficients on daily equity
returns are estimated with enormous standard errors and wander with any sample
you hand them, a six-month vintage is a second-order concern next to the
first-order error of forecasting a flat line.

*Why this is leak-free.* ``returns_from_features`` shifts ``ret_lag_1`` back by
one row within each ticker, so row ``t`` of the recovered panel holds the return
realised **on** day ``t`` - information available at the close of ``t``, never
later. The Kalman filter is a forward recursion, so its prediction for ``t``
depends only on observations strictly before ``t``. Perturbing any row after
``t`` therefore cannot move the prediction at ``t``; this is asserted directly in
``tests/test_arima_garch.py`` and generically in ``tests/test_models.py``.

Two smaller decisions
---------------------
*Feature-space rescaling.* ``predict`` is handed the **scaled** feature frame, so
``ret_lag_1`` - and hence the return series recovered from it - is in z-score
units rather than log-return units. Splicing a z-scored series onto raw training
returns would put a factor-of-eighty discontinuity in the middle of the endog.
The fix is an affine recalibration ``r = loc + scale * z`` whose two constants
are estimated by moment-matching **on training rows only** (the training targets
against the returns recovered from the training features), and therefore leak
nothing. When the frame is not scaled at all the calibration collapses to the
identity, so unscaled callers are unaffected.

*Horizon.* The model is designed for ``features.horizon == 1``, where the target
``y`` is exactly the one-day log return and the splice above is exact. For
``h > 1`` the target is an overlapping ``h``-day cumulative return while the
recovered extension is still a one-day return affinely rescaled to the target's
training moments - a documented approximation, not an exact identity.

Order selection
---------------
With ``auto = False`` the configured ``order`` is used as-is (``d`` must be 0;
the model is fitted on returns, which are already differenced prices). With
``auto = True`` the order is chosen per ticker by minimising the AIC over the
small grid ``p in 0..max_p`` and ``q in 0..max_q`` with ``d`` fixed at 0 -
``(max_p + 1) * (max_q + 1)`` candidate fits, 9 with the default configuration.
Candidates that raise or fail to converge are skipped; ties go to the first
candidate in ``p``-then-``q`` order, i.e. to the more parsimonious model.

Failure handling
----------------
A per-ticker fit that raises, fails to converge or returns non-finite parameters
does not abort the run: the ticker falls back to its training mean return, and a
warning naming the ticker is logged so the degradation is visible in the run log
rather than silently absorbed into the metrics.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from equity_forecast.config import ArimaConfig
from equity_forecast.models.base import BaseForecaster, register_model, returns_from_features
from equity_forecast.models.naive import _require_fitted

__all__ = ["ArimaForecaster"]

LOGGER = logging.getLogger(__name__)

#: Minimum number of finite training observations before a per-ticker fit is attempted.
MIN_TRAIN_OBS = 20

#: Minimum number of paired observations before the feature-space calibration is estimated.
MIN_CALIBRATION_OBS = 10

#: Relative standard deviation below which a training series counts as constant.
DEGENERATE_STD = 1e-12

_VALID_TRENDS = frozenset({"n", "c", "t", "ct"})


@dataclass(frozen=True)
class _Calibration:
    """An affine map from feature-space returns back to target units.

    Attributes:
        loc: Additive constant (the training mean the scaler removed).
        scale: Multiplicative constant (the training standard deviation the
            scaler divided by).
    """

    loc: float
    scale: float

    def apply(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map feature-space values into target units.

        Args:
            values: Returns as recovered from the (possibly scaled) feature frame.

        Returns:
            ``loc + scale * values``.
        """
        return np.asarray(self.loc + self.scale * values, dtype=np.float64)


IDENTITY_CALIBRATION = _Calibration(loc=0.0, scale=1.0)


def _finite(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return only the finite entries of an array.

    Args:
        values: Any float array.

    Returns:
        A 1-D array holding the finite entries, in order.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    return np.asarray(flat[np.isfinite(flat)], dtype=np.float64)


def _is_degenerate(values: npt.NDArray[np.float64]) -> bool:
    """Report whether a series has too little variation to fit anything to.

    The comparison is relative to the series' own magnitude, because a constant
    series does not always have an *exactly* zero sample standard deviation: the
    mean of ``n`` identical float64 values need not round-trip to that value, and
    the residual noise is around 1e-19. An absolute ``sd > 0`` test would let
    such a series through to an optimiser that cannot possibly fit it.

    Args:
        values: A ticker's finite training returns.

    Returns:
        ``True`` if the series is empty, non-finite, or numerically constant.
    """
    if values.size == 0:
        return True
    sd = float(values.std())
    if not np.isfinite(sd):
        return True
    scale = max(1.0, float(np.abs(values).max()))
    return sd <= DEGENERATE_STD * scale


def _fit_calibration(recovered: pd.Series, target: pd.Series) -> _Calibration:
    """Estimate the affine map from feature-space returns to target units.

    Both moments come from training rows only, so the map carries no information
    about the test block. A degenerate or too-short sample yields the identity,
    which is also the answer when the caller passes an unscaled feature frame.

    Args:
        recovered: Returns recovered from the *training* feature rows.
        target: The matching training targets.

    Returns:
        The fitted :class:`_Calibration`.
    """
    z = _finite(np.asarray(recovered.to_numpy(), dtype=np.float64))
    t = _finite(np.asarray(target.to_numpy(), dtype=np.float64))
    if z.size < MIN_CALIBRATION_OBS or t.size < MIN_CALIBRATION_OBS:
        return IDENTITY_CALIBRATION
    z_sd = float(z.std(ddof=1))
    t_sd = float(t.std(ddof=1))
    if not (np.isfinite(z_sd) and np.isfinite(t_sd)) or z_sd <= 0.0:
        return IDENTITY_CALIBRATION
    scale = t_sd / z_sd
    return _Calibration(loc=float(t.mean()) - scale * float(z.mean()), scale=scale)


def _ticker_positions(index: pd.Index) -> dict[str, npt.NDArray[np.intp]]:
    """Group row positions of a dataset index by ticker.

    Args:
        index: A ``(date, ticker)`` MultiIndex.

    Returns:
        A mapping from ticker to the positions of its rows, in index order.

    Raises:
        KeyError: If ``index`` has no ``ticker`` level.
    """
    labels = index.get_level_values("ticker")
    return {str(ticker): np.flatnonzero(np.asarray(labels == ticker)) for ticker in labels.unique()}


def _ticker_training_series(y: pd.Series) -> dict[str, pd.Series]:
    """Split the training target into one date-indexed series per ticker.

    Args:
        y: Training targets indexed by ``(date, ticker)``.

    Returns:
        A mapping from ticker to its finite training targets, sorted by date.

    Raises:
        KeyError: If ``y`` has no ``ticker`` index level.
    """
    out: dict[str, pd.Series] = {}
    for ticker, block in y.groupby(level="ticker"):
        series = block.droplevel("ticker").astype("float64").sort_index()
        values = np.asarray(series.to_numpy(), dtype=np.float64)
        out[str(ticker)] = series[np.isfinite(values)]
    return out


def _recovered_returns(
    panel: pd.DataFrame,
    ticker: str,
    dates: pd.Index,
    calibration: _Calibration,
) -> pd.Series:
    """Pull one ticker's recovered returns out of a return panel, in target units.

    Args:
        panel: Output of :func:`~equity_forecast.models.base.returns_from_features`.
        ticker: The ticker to extract.
        dates: The dates wanted, in any order.
        calibration: Affine map back to target units.

    Returns:
        A float64 series indexed by ``dates``; ``NaN`` where the return is
        genuinely unknown (notably the ticker's final row).
    """
    if ticker in panel.columns:
        raw = panel[ticker].reindex(dates)
    else:
        raw = pd.Series(np.nan, index=dates, dtype="float64")
    values = calibration.apply(np.asarray(raw.to_numpy(), dtype=np.float64))
    return pd.Series(values, index=dates, dtype="float64")


def _window_endog(
    train: pd.Series,
    recovered: pd.Series,
    dates: pd.Index,
    fill: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], int]:
    """Splice recovered window returns onto the training returns.

    The training block supplies the observations it knows (they *are* the fitted
    sample); every remaining date in the window is filled from the returns
    recovered from the feature frame. Non-finite entries - above all the final
    row of the window, whose return is genuinely unknown when its forecast has to
    be made - are replaced by the training mean. That substitution cannot leak:
    a forward filter's prediction for ``t`` never reads the observation at ``t``,
    so only rows *after* the substituted one could see it, and for the final row
    there are none.

    Args:
        train: The ticker's training targets, indexed by date.
        recovered: The ticker's window returns in target units, indexed by date.
        dates: The window dates for this ticker, in feature-frame row order.
        fill: Value substituted for non-finite observations (the training mean).

    Returns:
        A tuple of the spliced observation vector, the position of each entry of
        ``dates`` within it, and the number of substituted observations.
    """
    all_dates = train.index.union(dates)
    combined = pd.Series(np.nan, index=all_dates, dtype="float64")
    if len(recovered) > 0:
        combined.loc[recovered.index] = np.asarray(recovered.to_numpy(), dtype=np.float64)
    if len(train) > 0:
        combined.loc[train.index] = np.asarray(train.to_numpy(), dtype=np.float64)
    values = np.asarray(combined.to_numpy(), dtype=np.float64).copy()
    bad = ~np.isfinite(values)
    n_bad = int(bad.sum())
    if n_bad:
        values[bad] = fill
    positions = np.asarray(all_dates.get_indexer(dates), dtype=np.intp)
    return values, positions, n_bad


def _mean_or(values: npt.NDArray[np.float64], default: float) -> float:
    """Return the mean of the finite entries, or ``default`` if there are none.

    Args:
        values: Any float array.
        default: Fallback when nothing is finite.

    Returns:
        The finite mean, or ``default``.
    """
    finite = _finite(values)
    return float(finite.mean()) if finite.size else default


@register_model("arima")
class ArimaForecaster(BaseForecaster):
    """ARIMA(p, 0, q) fitted per ticker on returns, applied one step at a time.

    The model is estimated once on each fold's training returns and then
    re-filtered over the fold's full window with its parameters held fixed, which
    yields a genuine one-step-ahead forecast for every row rather than a
    multi-step forecast that decays to the unconditional mean. See the module
    docstring for the full rationale, the trade-off and the leak-freeness
    argument.

    Attributes:
        params: The :class:`~equity_forecast.config.ArimaConfig` in force.
        seed: Random seed; unused, maximum likelihood estimation here is
            deterministic.
    """

    def __init__(self, params: ArimaConfig, seed: int) -> None:
        """Store and validate configuration; no fitting happens here.

        Args:
            params: ARIMA hyperparameters (``order``, ``auto``, ``max_p``,
                ``max_q``, ``trend``).
            seed: Random seed; recorded for interface symmetry only.

        Raises:
            ValueError: If the order is not a triple, if ``d != 0`` (the model is
                fitted on returns, which are already differenced), if ``p`` or
                ``q`` is negative, if the auto grid bounds are negative, or if
                ``trend`` is not one of ``n``, ``c``, ``t``, ``ct``.
        """
        order = tuple(params.order)
        if len(order) != 3:
            msg = f"arima.order must be a (p, d, q) triple, got {params.order!r}"
            raise ValueError(msg)
        if order[1] != 0:
            msg = (
                f"arima.order must have d = 0 (the model is fitted on returns, which are "
                f"already the first difference of log prices), got d = {order[1]}"
            )
            raise ValueError(msg)
        if order[0] < 0 or order[2] < 0:
            msg = f"arima.order must have non-negative p and q, got {order!r}"
            raise ValueError(msg)
        if params.max_p < 0 or params.max_q < 0:
            msg = f"arima.max_p and arima.max_q must be >= 0, got {params.max_p}, {params.max_q}"
            raise ValueError(msg)
        if params.trend not in _VALID_TRENDS:
            msg = f"arima.trend must be one of {sorted(_VALID_TRENDS)}, got {params.trend!r}"
            raise ValueError(msg)

        self.params = params
        self.seed = seed
        self._order: tuple[int, int, int] = (int(order[0]), 0, int(order[2]))
        self._results: dict[str, Any] = {}
        self._orders: dict[str, tuple[int, int, int]] = {}
        self._train_series: dict[str, pd.Series] = {}
        self._calibrations: dict[str, _Calibration] = {}
        self._ticker_means: dict[str, float] = {}
        self._pooled_mean = 0.0
        self._fitted = False

    # ------------------------------------------------------------------ fit --

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ArimaForecaster:
        """Estimate one ARIMA per ticker on the fold's training returns.

        Args:
            x: Training feature frame indexed by ``(date, ticker)``; used for the
                ``ret_lag_1`` column, from which the feature-space calibration is
                estimated.
            y: Training targets indexed by ``(date, ticker)``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``y`` is empty.
            KeyError: If ``x`` has no ``ret_lag_1`` column or the index lacks a
                ``ticker`` level.
        """
        if len(y) == 0:
            msg = f"{self.name}.fit() got an empty training target"
            raise ValueError(msg)

        panel = returns_from_features(x)
        self._train_series = _ticker_training_series(y)
        self._pooled_mean = _mean_or(np.asarray(y.to_numpy(), dtype=np.float64), 0.0)
        self._results = {}
        self._orders = {}
        self._calibrations = {}
        self._ticker_means = {}

        for ticker, series in self._train_series.items():
            values = np.asarray(series.to_numpy(), dtype=np.float64)
            self._ticker_means[ticker] = _mean_or(values, self._pooled_mean)
            self._calibrations[ticker] = _fit_calibration(
                _recovered_returns(panel, ticker, series.index, IDENTITY_CALIBRATION),
                series,
            )
            fitted = self._fit_ticker(ticker, values)
            if fitted is not None:
                self._results[ticker] = fitted

        if not self._results:
            LOGGER.warning(
                "arima: no ticker produced a usable fit; every forecast will be a training mean"
            )
        self._fitted = True
        return self

    def _fit_ticker(
        self,
        ticker: str,
        values: npt.NDArray[np.float64],
    ) -> Any | None:  # noqa: ANN401
        """Fit one ticker, returning ``None`` if the ticker must fall back.

        Args:
            ticker: Ticker name, used in log messages.
            values: The ticker's finite training returns.

        Returns:
            The fitted statsmodels results object, or ``None`` if the series is
            unusable or no candidate order converged.
        """
        if values.size < MIN_TRAIN_OBS:
            LOGGER.warning(
                "arima: %s has only %d training observations (< %d); "
                "falling back to its training mean",
                ticker,
                values.size,
                MIN_TRAIN_OBS,
            )
            return None
        if _is_degenerate(values):
            LOGGER.warning(
                "arima: %s has a degenerate (numerically constant) training series; "
                "falling back to its training mean",
                ticker,
            )
            return None

        orders = self._candidate_orders()
        best: Any | None = None
        best_order = self._order
        best_aic = np.inf
        for order in orders:
            result = _estimate(values, order, self.params.trend)
            if result is None:
                continue
            aic = float(result.aic)
            if np.isfinite(aic) and aic < best_aic:
                best, best_order, best_aic = result, order, aic

        if best is None:
            LOGGER.warning(
                "arima: no candidate order converged for %s (tried %s); "
                "falling back to its training mean",
                ticker,
                [tuple(o) for o in orders],
            )
            return None
        self._orders[ticker] = best_order
        if self.params.auto:
            LOGGER.info("arima: selected order %s for %s (aic=%.2f)", best_order, ticker, best_aic)
        return best

    def _candidate_orders(self) -> list[tuple[int, int, int]]:
        """Return the orders to try for one ticker.

        Returns:
            ``[order]`` when ``auto`` is off, otherwise the full grid
            ``p in 0..max_p`` x ``q in 0..max_q`` with ``d = 0``, ordered by
            ``p`` then ``q`` so that ties in AIC go to the simpler model.
        """
        if not self.params.auto:
            return [self._order]
        return [
            (p, 0, q) for p in range(self.params.max_p + 1) for q in range(self.params.max_q + 1)
        ]

    # -------------------------------------------------------------- predict --

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Produce a one-step-ahead forecast for every row of the fold window.

        Args:
            x: Feature frame for the fold's full window (train + gap + test),
                indexed by ``(date, ticker)``.

        Returns:
            A ``float64`` series indexed exactly like ``x`` and named ``arima``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            KeyError: If ``x`` has no ``ret_lag_1`` column or the index lacks a
                ``ticker`` level.
        """
        _require_fitted(self._fitted, self.name)
        panel = returns_from_features(x)
        out = np.empty(len(x), dtype=np.float64)
        dates = x.index.get_level_values("date")

        for ticker, rows in _ticker_positions(x.index).items():
            fallback = self._ticker_means.get(ticker, self._pooled_mean)
            result = self._results.get(ticker)
            if result is None:
                out[rows] = fallback
                continue
            out[rows] = self._predict_ticker(ticker, result, panel, dates[rows], fallback)
        return pd.Series(out, index=x.index, name=self.name, dtype="float64")

    def _predict_ticker(
        self,
        ticker: str,
        result: Any,  # noqa: ANN401
        panel: pd.DataFrame,
        dates: pd.Index,
        fallback: float,
    ) -> npt.NDArray[np.float64]:
        """Re-filter one ticker's fitted model over the window and read it off.

        Args:
            ticker: Ticker name, used in log messages.
            result: The statsmodels results object fitted on the training block.
            panel: Recovered return panel for the whole window.
            dates: The ticker's window dates, in feature-frame row order.
            fallback: Value used wherever a forecast is not available.

        Returns:
            One forecast per entry of ``dates``, in the same order.
        """
        recovered = _recovered_returns(
            panel, ticker, dates, self._calibrations.get(ticker, IDENTITY_CALIBRATION)
        )
        train = self._train_series.get(ticker, pd.Series(dtype="float64"))
        values, positions, _ = _window_endog(train, recovered, dates, fallback)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                applied = result.apply(values, refit=False)
                predicted = applied.get_prediction(start=0, end=len(values) - 1).predicted_mean
        except Exception:  # one bad ticker must not abort the run
            LOGGER.warning(
                "arima: re-applying the fitted parameters failed for %s; "
                "falling back to its training mean",
                ticker,
                exc_info=True,
            )
            return np.full(len(dates), fallback, dtype=np.float64)

        picked = np.asarray(predicted, dtype=np.float64)[positions]
        bad = ~np.isfinite(picked)
        if bad.any():
            LOGGER.warning(
                "arima: %d non-finite forecast(s) for %s replaced by its training mean",
                int(bad.sum()),
                ticker,
            )
            picked[bad] = fallback
        return picked

    def predict_vol(self, x: pd.DataFrame) -> pd.Series | None:
        """Return ``None``: a homoskedastic ARIMA has no volatility forecast.

        The Gaussian ARIMA fitted here assumes a constant innovation variance, so
        the only "volatility forecast" it could offer is that constant - which
        would be worse than useless for vol targeting. Use ``arima_garch`` when a
        conditional variance is wanted.

        Args:
            x: Feature frame; ignored.

        Returns:
            ``None``.
        """
        del x
        return None

    # ----------------------------------------------------------- inspection --

    @property
    def selected_orders_(self) -> dict[str, tuple[int, int, int]]:
        """The ARIMA order actually used for each ticker that fitted.

        Returns:
            A mapping from ticker to its ``(p, d, q)`` order; tickers that fell
            back to their training mean are absent.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        return dict(self._orders)

    @property
    def fallback_tickers_(self) -> list[str]:
        """Tickers whose fit failed and which therefore predict their mean.

        Returns:
            The sorted names of tickers seen in training that have no usable fit.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        return sorted(set(self._train_series) - set(self._results))


def _estimate(
    values: npt.NDArray[np.float64],
    order: tuple[int, int, int],
    trend: str,
) -> Any | None:  # noqa: ANN401
    """Fit one ARIMA specification, returning ``None`` if it is unusable.

    Args:
        values: The training return series.
        order: ``(p, d, q)``, with ``d == 0``.
        trend: statsmodels trend specification.

    Returns:
        The fitted results object, or ``None`` if estimation raised, failed to
        converge, or produced non-finite parameters.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ARIMA(values, order=order, trend=trend).fit()
    except Exception:  # a failed order is skipped, not fatal
        LOGGER.debug("arima: order %s failed to estimate", order, exc_info=True)
        return None

    params = np.asarray(result.params, dtype=np.float64)
    if not np.isfinite(params).all():
        LOGGER.debug("arima: order %s produced non-finite parameters", order)
        return None
    retvals = getattr(result, "mle_retvals", None) or {}
    if not retvals.get("converged", True):
        LOGGER.debug("arima: order %s did not converge", order)
        return None
    return result
