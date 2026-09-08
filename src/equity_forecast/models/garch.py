"""ARX mean model with GARCH errors: a conditional mean *and* a conditional variance.

What this model is for
----------------------
GARCH is a **variance** model. It says nothing new about where returns are
going; it says how uncertain tomorrow's return is, given how turbulent the last
few weeks have been. The single most common abuse of it in equity forecasting is
to plot the conditional variance next to the price and present it as a
prediction - it is not, and this class refuses to pretend otherwise:

* :meth:`ArimaGarchForecaster.predict` returns the conditional **mean** from the
  ARX (autoregressive) part of the model. That is the only object here that is
  comparable with the other models' point forecasts.
* :meth:`ArimaGarchForecaster.predict_vol` returns the one-step-ahead
  conditional **standard deviation** of the daily log return - a positive number
  of order 0.005-0.05 for equities. It feeds volatility targeting in the
  backtest, where knowing the size of tomorrow's move is genuinely valuable even
  when its sign is unknowable.

Estimating the mean jointly with the variance is also the point of using ``arch``
rather than fitting an ARIMA and a GARCH separately: the AR coefficients are
estimated by weighting each observation by its conditional precision, so quiet
periods (where the signal-to-noise ratio is highest) count for more than crisis
periods, and the AR parameters are no longer dragged around by a handful of
outliers.

Design decisions inherited from the ARIMA module
------------------------------------------------
This model uses exactly the same fixed-parameters-per-fold, genuinely
one-step-ahead scheme as :mod:`equity_forecast.models.arima`, and reuses that
module's splicing helpers (hence the import): parameters are estimated once on
the fold's training returns, the realised return series over the prediction
window is recovered with
:func:`equity_forecast.models.base.returns_from_features` and spliced onto the
training returns, and the *fixed* parameters are then run forward over the
extended series. Read that module's docstring for the full argument, including
why the scheme is leak-free and what it costs. The affine feature-space
recalibration described there is applied here too.

Two things are specific to this module.

*Percent scaling.* The ``arch`` optimiser is written for data of order 1, not of
order 0.01. Handed raw daily log returns it either warns and rescales behind your
back or converges to a boundary solution with an ``omega`` around 1e-6 that the
line search cannot resolve. Every series is therefore multiplied by 100 before
fitting (``rescale=False`` then keeps ``arch`` from doing anything further of its
own), and the conditional mean and volatility are divided by 100 on the way out,
so the numbers this class returns are always in log-return units.

*How the fixed parameters are re-run.* The obvious route - build a new
:class:`~arch.univariate.ARX` on the extended series and call ``.fix(params)`` -
was implemented and tested first, and it produces the right numbers but is **not
strictly causal**: ``arch`` derives the variance bounds it clamps the recursion
with from ``np.var(resids)`` and ``np.max(resids**2)`` over the *whole* sample
(see ``VolatilityProcess.variance_bounds``), so perturbing the last few rows of
the window perceptibly moves the conditional volatility of earlier rows. That
would fail the causality test, and rightly so. What this module does instead is
drive the recursion itself: the ARX conditional mean is evaluated directly from
the fitted coefficients, and the resulting residuals are fed to ``arch``'s own
tested ``GARCH.compute_variance`` with a backcast and (constant) variance bounds
computed **once, from the training residuals**. Nothing in the recursion then
depends on anything but the fitted parameters and the past, and the output
matches ``.fix()`` to floating-point precision once the warm-up has decayed.

Failure handling
----------------
A ticker whose fit raises, reports a non-zero ``arch`` convergence flag, or
returns non-finite parameters falls back to its training mean for the point
forecast and its training standard deviation for the volatility, with a warning
naming the ticker. The volatility is floored at a tiny positive number so that
``predict_vol`` is strictly positive even for a degenerate constant series.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from arch.univariate import ARX, GARCH, GeneralizedError, Normal, SkewStudent, StudentsT

from equity_forecast.config import ArimaGarchConfig
from equity_forecast.models.arima import (
    DEGENERATE_STD,
    IDENTITY_CALIBRATION,
    MIN_TRAIN_OBS,
    _Calibration,
    _fit_calibration,
    _is_degenerate,
    _mean_or,
    _recovered_returns,
    _ticker_positions,
    _ticker_training_series,
    _window_endog,
)
from equity_forecast.models.base import BaseForecaster, register_model, returns_from_features
from equity_forecast.models.naive import _require_fitted

__all__ = ["ArimaGarchForecaster"]

LOGGER = logging.getLogger(__name__)

#: ``arch`` is numerically much better behaved on returns expressed in percent.
PERCENT = 100.0

#: Floor applied to every volatility forecast, so ``predict_vol`` is strictly positive.
MIN_VOL = 1e-8

#: Supported ``dist`` values, mapped to the ``arch`` distribution they select.
DISTRIBUTIONS: dict[str, type[Any]] = {
    "normal": Normal,
    "gaussian": Normal,
    "t": StudentsT,
    "studentst": StudentsT,
    "skewt": SkewStudent,
    "skewstudent": SkewStudent,
    "ged": GeneralizedError,
}


@dataclass(frozen=True)
class _GarchFit:
    """Everything needed to re-run a fitted model forward, and nothing else.

    Attributes:
        mean_params: ARX mean parameters in percent units: the constant followed
            by one coefficient per lag.
        vol_params: GARCH parameters in percent units: ``omega`` followed by the
            ARCH and GARCH coefficients, in ``arch``'s own order.
        lags: Number of autoregressive lags in the mean equation.
        backcast: Variance used to seed the recursion, computed from the training
            residuals only.
        var_lower: Lower clamp on the conditional variance (training-derived).
        var_upper: Upper clamp on the conditional variance (training-derived).
        warmup_mean: Conditional mean, in percent units, assigned to the first
            ``lags`` rows, for which the AR recursion is not yet defined.
    """

    mean_params: npt.NDArray[np.float64]
    vol_params: npt.NDArray[np.float64]
    lags: int
    backcast: float
    var_lower: float
    var_upper: float
    warmup_mean: float


def _conditional_mean(
    values: npt.NDArray[np.float64], mean_params: npt.NDArray[np.float64], lags: int, warmup: float
) -> npt.NDArray[np.float64]:
    """Evaluate the ARX conditional mean at every observation.

    ``mu_t = c + sum_i phi_i * y_{t-i}`` reads only strictly past observations,
    which is what makes the whole scheme causal. The first ``lags`` rows have no
    complete history and are given the training mean instead; they sit at the
    very start of the fold's training block, hundreds of rows before anything is
    scored.

    Args:
        values: The observation series in percent units.
        mean_params: Constant followed by one coefficient per lag.
        lags: Number of autoregressive lags.
        warmup: Value used for the first ``lags`` rows.

    Returns:
        The conditional mean at every observation, in percent units.
    """
    n = values.size
    mu = np.full(n, warmup, dtype=np.float64)
    if n <= lags:
        return mu
    acc = np.full(n - lags, float(mean_params[0]), dtype=np.float64)
    for i in range(1, lags + 1):
        acc += float(mean_params[i]) * values[lags - i : n - i]
    mu[lags:] = acc
    return mu


def _conditional_moments(
    fit: _GarchFit,
    values: npt.NDArray[np.float64],
    vol: Any,  # noqa: ANN401
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the fitted model forward over a series, in percent units.

    Args:
        fit: The parameters and initial conditions captured at fit time.
        values: The observation series in percent units, free of ``NaN``.
        vol: The :class:`~arch.univariate.GARCH` volatility process to recurse with.

    Returns:
        A tuple of the conditional mean and the conditional standard deviation at
        every observation, both in percent units.
    """
    mu = _conditional_mean(values, fit.mean_params, fit.lags, fit.warmup_mean)
    resid = values - mu
    n = values.size
    var_bounds = np.empty((n, 2), dtype=np.float64)
    var_bounds[:, 0] = fit.var_lower
    var_bounds[:, 1] = fit.var_upper
    sigma2 = np.zeros(n, dtype=np.float64)
    vol.compute_variance(fit.vol_params, resid, sigma2, fit.backcast, var_bounds)
    return mu, np.sqrt(np.asarray(sigma2, dtype=np.float64))


@register_model("arima_garch")
class ArimaGarchForecaster(BaseForecaster):
    """ARX(``mean_lags``) conditional mean with GARCH(p, q) conditional variance.

    Fitted once per ticker per fold on the training returns and then re-run
    forward over the fold's whole window with its parameters fixed, so that both
    the mean and the volatility it reports for row ``t`` condition on returns up
    to ``t-1`` only. See the module docstring for the reasoning and for the
    percent-scaling convention.

    Attributes:
        params: The :class:`~equity_forecast.config.ArimaGarchConfig` in force.
        seed: Random seed; unused, estimation here is deterministic.
    """

    def __init__(self, params: ArimaGarchConfig, seed: int) -> None:
        """Store and validate configuration; no fitting happens here.

        Args:
            params: Mean-lag count, error distribution and GARCH orders.
            seed: Random seed; recorded for interface symmetry only.

        Raises:
            ValueError: If ``mean_lags`` or ``q`` is negative, if ``p < 1`` (a
                GARCH model needs at least one ARCH term), or if ``dist`` is not
                one of the supported distributions.
        """
        if params.mean_lags < 0:
            msg = f"arima_garch.mean_lags must be >= 0, got {params.mean_lags}"
            raise ValueError(msg)
        if params.p < 1:
            msg = f"arima_garch.p must be >= 1 (a GARCH needs an ARCH term), got {params.p}"
            raise ValueError(msg)
        if params.q < 0:
            msg = f"arima_garch.q must be >= 0, got {params.q}"
            raise ValueError(msg)
        key = params.dist.strip().lower()
        if key not in DISTRIBUTIONS:
            msg = f"arima_garch.dist must be one of {sorted(DISTRIBUTIONS)}, got {params.dist!r}"
            raise ValueError(msg)

        self.params = params
        self.seed = seed
        self._distribution: type[Any] = DISTRIBUTIONS[key]
        self._fits: dict[str, _GarchFit] = {}
        self._train_series: dict[str, pd.Series] = {}
        self._calibrations: dict[str, _Calibration] = {}
        self._ticker_means: dict[str, float] = {}
        self._ticker_vols: dict[str, float] = {}
        self._pooled_mean = 0.0
        self._pooled_vol = MIN_VOL
        self._fitted = False

    def _volatility(self) -> Any:  # noqa: ANN401 - `arch` ships no useful public types
        """Return a fresh GARCH volatility process matching the configuration.

        Returns:
            A :class:`~arch.univariate.GARCH` with the configured ``p`` and ``q``
            and no asymmetry term.
        """
        return GARCH(p=self.params.p, q=self.params.q)

    # ------------------------------------------------------------------ fit --

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ArimaGarchForecaster:
        """Estimate one ARX-GARCH per ticker on the fold's training returns.

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
        pooled = np.asarray(y.to_numpy(), dtype=np.float64)
        self._pooled_mean = _mean_or(pooled, 0.0)
        self._pooled_vol = _positive_std(pooled, MIN_VOL)
        self._fits = {}
        self._calibrations = {}
        self._ticker_means = {}
        self._ticker_vols = {}

        for ticker, series in self._train_series.items():
            values = np.asarray(series.to_numpy(), dtype=np.float64)
            self._ticker_means[ticker] = _mean_or(values, self._pooled_mean)
            self._ticker_vols[ticker] = _positive_std(values, self._pooled_vol)
            self._calibrations[ticker] = _fit_calibration(
                _recovered_returns(panel, ticker, series.index, IDENTITY_CALIBRATION),
                series,
            )
            fitted = self._fit_ticker(ticker, values)
            if fitted is not None:
                self._fits[ticker] = fitted

        if not self._fits:
            LOGGER.warning(
                "arima_garch: no ticker produced a usable fit; every forecast will be a "
                "training mean and every volatility a training standard deviation"
            )
        self._fitted = True
        return self

    def _fit_ticker(self, ticker: str, values: npt.NDArray[np.float64]) -> _GarchFit | None:
        """Fit one ticker, returning ``None`` if the ticker must fall back.

        Args:
            ticker: Ticker name, used in log messages.
            values: The ticker's finite training returns, in log-return units.

        Returns:
            The captured :class:`_GarchFit`, or ``None`` if the series is
            degenerate or estimation failed.
        """
        if values.size < MIN_TRAIN_OBS:
            LOGGER.warning(
                "arima_garch: %s has only %d training observations (< %d); falling back to "
                "its training mean and standard deviation",
                ticker,
                values.size,
                MIN_TRAIN_OBS,
            )
            return None
        if _is_degenerate(values):
            LOGGER.warning(
                "arima_garch: %s has a degenerate (numerically constant) training series; "
                "falling back to its training mean and standard deviation",
                ticker,
            )
            return None

        scaled = values * PERCENT
        lags = self.params.mean_lags
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARX(
                    scaled,
                    lags=lags,
                    constant=True,
                    volatility=self._volatility(),
                    distribution=self._distribution(),
                    rescale=False,
                )
                result = model.fit(disp="off", show_warning=False)
        except Exception:  # one bad ticker must not abort the run
            LOGGER.warning(
                "arima_garch: estimation raised for %s; falling back to its training mean "
                "and standard deviation",
                ticker,
                exc_info=True,
            )
            return None

        params = np.asarray(result.params, dtype=np.float64)
        flag = int(getattr(result, "convergence_flag", 0))
        if not np.isfinite(params).all() or flag != 0:
            LOGGER.warning(
                "arima_garch: %s failed to converge (flag=%d); falling back to its training "
                "mean and standard deviation",
                ticker,
                flag,
            )
            return None
        return self._capture(model, params, scaled)

    def _capture(
        self,
        model: Any,  # noqa: ANN401
        params: npt.NDArray[np.float64],
        scaled: npt.NDArray[np.float64],
    ) -> _GarchFit:
        """Freeze a fitted model into the parameters needed to re-run it.

        The backcast and the variance bounds are computed here, from the
        *training* residuals only, so that re-running the recursion over a longer
        series later cannot let the future influence the past.

        Args:
            model: The fitted :class:`~arch.univariate.ARX` model.
            params: Its full parameter vector (mean, then volatility, then
                distribution).
            scaled: The training returns in percent units.

        Returns:
            The captured :class:`_GarchFit`.
        """
        n_mean = int(model.num_params)
        n_vol = int(model.volatility.num_params)
        mean_params = params[:n_mean]
        vol_params = params[n_mean : n_mean + n_vol]
        lags = self.params.mean_lags
        warmup = float(scaled.mean())

        resid = scaled - _conditional_mean(scaled, mean_params, lags, warmup)
        vol = model.volatility
        bounds = np.asarray(vol.variance_bounds(resid), dtype=np.float64)
        return _GarchFit(
            mean_params=mean_params,
            vol_params=vol_params,
            lags=lags,
            backcast=float(np.asarray(vol.backcast(resid), dtype=np.float64).mean()),
            var_lower=float(bounds[:, 0].min()),
            var_upper=float(bounds[:, 1].max()),
            warmup_mean=warmup,
        )

    # -------------------------------------------------------------- predict --

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Forecast the conditional **mean** return for every row of the window.

        This is the ARX part of the model. The GARCH part describes the size of
        the errors around it, not the errors themselves, so it plays no role
        here - see :meth:`predict_vol`.

        Args:
            x: Feature frame for the fold's full window, indexed by
                ``(date, ticker)``.

        Returns:
            A ``float64`` series indexed exactly like ``x``, named
            ``arima_garch``, in log-return units.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            KeyError: If ``x`` has no ``ret_lag_1`` column or the index lacks a
                ``ticker`` level.
        """
        mean, _ = self._run(x)
        return pd.Series(mean, index=x.index, name=self.name, dtype="float64")

    def predict_vol(self, x: pd.DataFrame) -> pd.Series | None:
        """Forecast the one-step-ahead conditional volatility for every row.

        Args:
            x: Feature frame for the fold's full window, indexed by
                ``(date, ticker)``.

        Returns:
            A strictly positive ``float64`` series indexed exactly like ``x``,
            named ``arima_garch``, holding the daily standard deviation of the
            log return - typically 0.005-0.05 for equities.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            KeyError: If ``x`` has no ``ret_lag_1`` column or the index lacks a
                ``ticker`` level.
        """
        _, vol = self._run(x)
        return pd.Series(vol, index=x.index, name=self.name, dtype="float64")

    def _run(self, x: pd.DataFrame) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Run every ticker's fitted model forward over the window.

        Args:
            x: Feature frame for the fold's full window.

        Returns:
            The conditional mean and conditional volatility for every row of
            ``x``, in row order and in log-return units.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        _require_fitted(self._fitted, self.name)
        panel = returns_from_features(x)
        n = len(x)
        mean_out = np.empty(n, dtype=np.float64)
        vol_out = np.empty(n, dtype=np.float64)
        dates = x.index.get_level_values("date")
        volatility = self._volatility()

        for ticker, rows in _ticker_positions(x.index).items():
            mean_fallback = self._ticker_means.get(ticker, self._pooled_mean)
            vol_fallback = max(self._ticker_vols.get(ticker, self._pooled_vol), MIN_VOL)
            fit = self._fits.get(ticker)
            if fit is None:
                mean_out[rows] = mean_fallback
                vol_out[rows] = vol_fallback
                continue
            mean, vol = self._run_ticker(
                ticker,
                fit,
                volatility=volatility,
                panel=panel,
                dates=dates[rows],
                mean_fallback=mean_fallback,
                vol_fallback=vol_fallback,
            )
            mean_out[rows] = mean
            vol_out[rows] = vol
        return mean_out, vol_out

    def _run_ticker(
        self,
        ticker: str,
        fit: _GarchFit,
        *,
        volatility: Any,  # noqa: ANN401
        panel: pd.DataFrame,
        dates: pd.Index,
        mean_fallback: float,
        vol_fallback: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Re-run one ticker's fixed parameters over the fold window.

        Args:
            ticker: Ticker name, used in log messages.
            fit: The parameters captured at fit time.
            volatility: The GARCH process to recurse with.
            panel: Recovered return panel for the whole window.
            dates: The ticker's window dates, in feature-frame row order.
            mean_fallback: Value used wherever a mean forecast is unavailable.
            vol_fallback: Value used wherever a volatility forecast is unavailable.

        Returns:
            The conditional mean and volatility at each entry of ``dates``, in
            log-return units.
        """
        recovered = _recovered_returns(
            panel, ticker, dates, self._calibrations.get(ticker, IDENTITY_CALIBRATION)
        )
        train = self._train_series.get(ticker, pd.Series(dtype="float64"))
        values, positions, _ = _window_endog(train, recovered, dates, mean_fallback)
        try:
            mu, sigma = _conditional_moments(fit, values * PERCENT, volatility)
        except Exception:  # one bad ticker must not abort the run
            LOGGER.warning(
                "arima_garch: re-running the fitted parameters failed for %s; falling back "
                "to its training mean and standard deviation",
                ticker,
                exc_info=True,
            )
            n = len(dates)
            return (
                np.full(n, mean_fallback, dtype=np.float64),
                np.full(n, vol_fallback, dtype=np.float64),
            )

        mean = np.asarray(mu, dtype=np.float64)[positions] / PERCENT
        vol = np.asarray(sigma, dtype=np.float64)[positions] / PERCENT
        return (
            _repair(mean, mean_fallback, ticker, "mean forecast"),
            np.maximum(_repair(vol, vol_fallback, ticker, "volatility forecast"), MIN_VOL),
        )

    # ----------------------------------------------------------- inspection --

    @property
    def fallback_tickers_(self) -> list[str]:
        """Tickers whose fit failed and which therefore predict their moments.

        Returns:
            The sorted names of tickers seen in training that have no usable fit.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        return sorted(set(self._train_series) - set(self._fits))


def _positive_std(values: npt.NDArray[np.float64], default: float) -> float:
    """Return the standard deviation of the finite entries, floored positive.

    A numerically constant sample falls back to ``default`` (the pooled
    cross-sectional standard deviation) rather than to its own near-zero
    standard deviation: a broken ticker is better described by the panel it sits
    in than by a volatility of zero, which would blow up any vol-targeted
    position built on it.

    Args:
        values: Any float array.
        default: Fallback when the sample is too small or degenerate.

    Returns:
        A strictly positive standard deviation.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return max(default, MIN_VOL)
    sd = float(finite.std(ddof=1))
    scale = max(1.0, float(np.abs(finite).max()))
    if not np.isfinite(sd) or sd <= DEGENERATE_STD * scale:
        return max(default, MIN_VOL)
    return sd


def _repair(
    values: npt.NDArray[np.float64], fallback: float, ticker: str, what: str
) -> npt.NDArray[np.float64]:
    """Replace non-finite entries with a fallback, logging that it happened.

    Args:
        values: The forecasts to check.
        fallback: Replacement value.
        ticker: Ticker name, used in the log message.
        what: Human-readable name of the quantity, used in the log message.

    Returns:
        ``values`` with every non-finite entry replaced.
    """
    bad = ~np.isfinite(values)
    if bad.any():
        LOGGER.warning(
            "arima_garch: %d non-finite %s(s) for %s replaced by a training moment",
            int(bad.sum()),
            what,
            ticker,
        )
        values = values.copy()
        values[bad] = fallback
    return values
