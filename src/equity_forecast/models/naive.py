"""Return baselines: the bar every other model in this project has to clear.

Forecasting *price levels* is the classic trap in this problem. A daily close
series is very nearly a random walk, so the lazy predictor
``P_hat_t = P_{t-1}`` scores an out-of-sample R-squared above 0.99 and draws a
chart that hugs the truth beautifully - while containing exactly zero
information about what happens next. Every apparent success is just yesterday's
price echoing back one day late.

Differencing removes the illusion. Once the target is the log return
``r_t = log(P_t) - log(P_{t-1})``, that same random-walk forecast is no longer a
near-perfect curve but a flat line at zero, and its R-squared is roughly zero.
That is the honest starting point, and it is exactly what :class:`ZeroReturn`
implements: it predicts ``0.0`` for every row, forever. Any model whose error
metrics do not beat it has learned nothing, however good its price chart looks.

:class:`HistoricalMean` is one small step up: it predicts the average return
seen during training, per ticker where the ticker was in the training set and
pooled across the cross-section otherwise. It captures drift and nothing else,
which makes it the natural second reference point - beating zero merely by
being long a rising market is not a forecast.

Both baselines are trivially causal: their prediction for row ``t`` ignores the
contents of every row, so no future information can reach it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from equity_forecast.models.base import BaseForecaster, register_model

__all__ = [
    "HistoricalMean",
    "NotFittedError",
    "ZeroReturn",
]

LOGGER = logging.getLogger(__name__)


class NotFittedError(RuntimeError, ValueError):
    """Raised when ``predict`` is called on a forecaster that was never fitted.

    It inherits from both :class:`RuntimeError` (calling methods out of order is
    a state error) and :class:`ValueError` (scikit-learn's own ``NotFittedError``
    is a ``ValueError``), so callers can catch whichever they find natural.

    It lives in this module rather than in :mod:`equity_forecast.models.base`
    only because that module is frozen; every model in the package raises it.
    """


def _require_fitted(fitted: bool, model: str) -> None:
    """Raise if a model has not been fitted yet.

    Args:
        fitted: Whether ``fit`` has completed successfully.
        model: Model name, used in the error message.

    Raises:
        NotFittedError: If ``fitted`` is false.
    """
    if not fitted:
        msg = (
            f"{model}.predict() was called before fit(); "
            f"fit the model on the fold's training rows first"
        )
        raise NotFittedError(msg)


def _ticker_labels(index: pd.Index) -> npt.NDArray[np.object_]:
    """Extract the ``ticker`` level of a dataset index.

    Args:
        index: A ``(date, ticker)`` MultiIndex.

    Returns:
        The ticker label of every row, in index order.

    Raises:
        KeyError: If ``index`` has no ``ticker`` level.
    """
    try:
        level = index.get_level_values("ticker")
    except KeyError as exc:  # pragma: no cover - defensive, schema guarantees the level
        msg = f"expected a (date, ticker) MultiIndex with a 'ticker' level, got {index.names}"
        raise KeyError(msg) from exc
    return np.asarray(level.to_numpy(), dtype=object)


def _target_values(y: pd.Series, model: str) -> npt.NDArray[np.float64]:
    """Return the training target as a float array, rejecting an empty one.

    Args:
        y: Training targets for one fold.
        model: Model name, used in the error message.

    Returns:
        ``y`` as a ``float64`` array.

    Raises:
        ValueError: If ``y`` is empty.
    """
    if len(y) == 0:
        msg = f"{model}.fit() got an empty training target"
        raise ValueError(msg)
    return np.asarray(y.to_numpy(), dtype="float64")


@register_model("zero_return")
class ZeroReturn(BaseForecaster):
    """The random-walk baseline: predict a return of exactly zero, always.

    This is the forecast implied by "prices are a martingale". It has no
    parameters and learns nothing from the training set - ``fit`` only records
    that it was called - which is the point: it is the reference every other
    model's errors are measured against.
    """

    def __init__(self, params: dict[str, Any], seed: int) -> None:
        """Store configuration.

        Args:
            params: Configuration mapping; this model has no knobs, so any keys
                are ignored (and logged at debug level).
            seed: Random seed; unused, this model is deterministic.
        """
        self.params = dict(params)
        self.seed = seed
        self._fitted = False
        if self.params:
            LOGGER.debug("zero_return takes no parameters; ignoring %s", sorted(self.params))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ZeroReturn:
        """Record that the model has been fitted; nothing is estimated.

        Args:
            x: Training feature frame; ignored.
            y: Training targets; ignored.

        Returns:
            ``self``.
        """
        del x, y
        self._fitted = True
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict zero for every row of ``x``.

        Args:
            x: Feature frame for the fold's full window.

        Returns:
            A ``float64`` series of zeros indexed exactly like ``x``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        _require_fitted(self._fitted, self.name)
        return pd.Series(0.0, index=x.index, name=self.name, dtype="float64")


@register_model("historical_mean")
class HistoricalMean(BaseForecaster):
    """Predict the training-set mean return, per ticker where possible.

    The mean is estimated separately for every ticker seen during training, so a
    persistently drifting name gets its own drift. A ticker absent from the
    training set (or whose training returns are entirely missing) falls back to
    the pooled mean over all training rows, which is the only defensible guess
    when the model has never met the name.

    The estimate uses training rows only, so the prediction for row ``t`` is a
    constant that does not depend on any row of the frame handed to
    :meth:`predict` - causality is trivial.
    """

    def __init__(self, params: dict[str, Any], seed: int) -> None:
        """Store configuration.

        Args:
            params: Configuration mapping; this model has no knobs, so any keys
                are ignored (and logged at debug level).
            seed: Random seed; unused, this model is deterministic.
        """
        self.params = dict(params)
        self.seed = seed
        self._ticker_means: dict[str, float] = {}
        self._pooled_mean = 0.0
        self._fitted = False
        if self.params:
            LOGGER.debug("historical_mean takes no parameters; ignoring %s", sorted(self.params))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> HistoricalMean:
        """Estimate the pooled and per-ticker mean training return.

        Args:
            x: Training feature frame; ignored, only the target matters here.
            y: Training targets indexed by ``(date, ticker)``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``y`` is empty.
            KeyError: If ``y`` has no ``ticker`` index level.
        """
        del x
        values = _target_values(y, self.name)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            LOGGER.warning("historical_mean: no finite training targets; falling back to 0.0")
            self._pooled_mean = 0.0
        else:
            self._pooled_mean = float(finite.mean())

        labels = pd.Index(_ticker_labels(y.index), name="ticker")
        per_ticker = pd.Series(values, index=labels).groupby(level="ticker").mean()
        self._ticker_means = {
            str(ticker): float(mean) for ticker, mean in per_ticker.items() if np.isfinite(mean)
        }
        self._fitted = True
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict each row's ticker mean, falling back to the pooled mean.

        Args:
            x: Feature frame for the fold's full window, indexed by
                ``(date, ticker)``.

        Returns:
            A ``float64`` series indexed exactly like ``x``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            KeyError: If ``x`` has no ``ticker`` index level.
        """
        _require_fitted(self._fitted, self.name)
        labels = _ticker_labels(x.index)
        means = self._ticker_means
        pooled = self._pooled_mean
        values = np.fromiter(
            (means.get(str(ticker), pooled) for ticker in labels),
            dtype="float64",
            count=len(labels),
        )
        return pd.Series(values, index=x.index, name=self.name, dtype="float64")

    @property
    def pooled_mean_(self) -> float:
        """The pooled training mean return.

        Returns:
            The mean over all finite training targets.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        return self._pooled_mean

    @property
    def ticker_means_(self) -> pd.Series:
        """The per-ticker training mean returns.

        Returns:
            A ``float64`` series indexed by ticker, sorted by ticker.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        index = pd.Index(sorted(self._ticker_means), name="ticker")
        return pd.Series(
            [self._ticker_means[t] for t in index],
            index=index,
            name="mean_return",
            dtype="float64",
        )
