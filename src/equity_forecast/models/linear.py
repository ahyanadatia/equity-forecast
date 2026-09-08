"""Ridge regression: the linear model that actually uses the feature matrix.

The baselines in :mod:`equity_forecast.models.naive` ignore the features
entirely. Ridge is the first model that reads them, and it reads all of them at
once - lagged returns, rolling moments, drawdown, realised vol and the
day-of-week dummies - fitting a single set of coefficients.

Two design choices are worth stating plainly.

*Why the L2 penalty.* The feature matrix is badly collinear: ``roll_mean_5``
is a linear combination of the very lags sitting next to it, and realised vol
and rolling standard deviation measure nearly the same thing. Ordinary least
squares responds to collinearity with huge, unstable, sign-flipping
coefficients that fit the training fold's noise and fall apart out of sample.
The ridge penalty trades a little bias for a large reduction in variance, which
is the right trade when the signal-to-noise ratio is as low as it is in daily
equity returns. As ``alpha`` grows the fit degenerates gracefully towards the
intercept - that is, towards the historical-mean baseline.

*Why pooled across tickers.* One model is fitted on the stacked cross-section
rather than one per name. Any per-ticker relationship strong enough to survive
out of sample is worth more estimated on ``n_tickers`` times as much data, and
per-ticker fits on a few years of daily returns overfit almost immediately.

The fitted coefficients are exposed through :attr:`Ridge.coefficients_` so the
model can be reported on and argued with rather than treated as a black box.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import Ridge as SkRidge

from equity_forecast.config import RidgeConfig
from equity_forecast.models.base import BaseForecaster, register_model
from equity_forecast.models.naive import _require_fitted

__all__ = ["Ridge"]

LOGGER = logging.getLogger(__name__)


@register_model("ridge")
class Ridge(BaseForecaster):
    """L2-penalised linear regression of the target on the full feature matrix.

    The model expects the *scaled* feature frame (see
    :func:`equity_forecast.features.apply_scaler`): a single ``alpha`` only
    means the same thing for every coefficient if the columns share a scale.

    Attributes:
        params: The :class:`~equity_forecast.config.RidgeConfig` in force.
        seed: Random seed handed to scikit-learn (only some solvers use it).
    """

    def __init__(self, params: RidgeConfig, seed: int) -> None:
        """Store configuration; no fitting happens here.

        Args:
            params: Ridge hyperparameters (``alpha``, ``fit_intercept``).
            seed: Random seed, forwarded to scikit-learn for reproducibility.
        """
        self.params = params
        self.seed = seed
        self._model: Any = None
        self._columns: list[str] = []
        self._impute: npt.NDArray[np.float64] = np.zeros(0, dtype="float64")
        self._fitted = False

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Ridge:
        """Fit the ridge regression on the fold's training rows.

        Rows carrying a non-finite feature or target are dropped rather than
        propagated: a single ``NaN`` would otherwise poison every coefficient.
        The column means of the surviving training rows are remembered so that
        :meth:`predict` can impute non-finite cells with a statistic that is
        purely a function of the training fold, which keeps the model causal.

        Args:
            x: Scaled training feature frame indexed by ``(date, ticker)``.
            y: Training targets, aligned row-for-row with ``x``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``x`` has no columns or no rows, if ``y`` does not
                line up with ``x``, or if no training row is usable.
        """
        if x.shape[1] == 0:
            msg = "ridge.fit() got a feature frame with no columns"
            raise ValueError(msg)
        if len(x) == 0:
            msg = "ridge.fit() got an empty training set"
            raise ValueError(msg)
        if len(y) != len(x):
            msg = f"ridge.fit() got {len(x)} feature rows but {len(y)} targets"
            raise ValueError(msg)
        if not x.index.equals(y.index):
            LOGGER.debug("ridge.fit(): x and y indexes differ; aligning positionally")

        columns = [str(c) for c in x.columns]
        matrix = np.asarray(x.to_numpy(dtype="float64"), dtype="float64")
        target = np.asarray(y.to_numpy(), dtype="float64")

        usable = np.isfinite(matrix).all(axis=1) & np.isfinite(target)
        n_dropped = int(matrix.shape[0] - usable.sum())
        if n_dropped:
            LOGGER.debug(
                "ridge.fit(): dropping %d training row(s) with non-finite values", n_dropped
            )
        if not usable.any():
            msg = "ridge.fit() found no training row free of non-finite values"
            raise ValueError(msg)

        kept = matrix[usable]
        model = SkRidge(
            alpha=float(self.params.alpha),
            fit_intercept=bool(self.params.fit_intercept),
            random_state=self.seed,
        )
        model.fit(kept, target[usable])

        self._model = model
        self._columns = columns
        self._impute = np.asarray(kept.mean(axis=0), dtype="float64")
        self._fitted = True
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict the target for every row of ``x``.

        Args:
            x: Scaled feature frame for the fold's full window. It must contain
                every column the model was fitted on; extra columns are ignored.

        Returns:
            A ``float64`` series indexed exactly like ``x``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            ValueError: If a fitted feature column is missing from ``x``.
        """
        _require_fitted(self._fitted, self.name)
        missing = [c for c in self._columns if c not in x.columns]
        if missing:
            msg = f"ridge.predict() is missing feature column(s) seen during fit: {missing}"
            raise ValueError(msg)

        matrix = np.asarray(x[self._columns].to_numpy(dtype="float64"), dtype="float64").copy()
        bad = ~np.isfinite(matrix)
        if bad.any():
            LOGGER.debug("ridge.predict(): imputing %d non-finite cell(s)", int(bad.sum()))
            matrix[bad] = np.take(self._impute, np.nonzero(bad)[1])

        values = np.asarray(self._model.predict(matrix), dtype="float64").ravel()
        return pd.Series(values, index=x.index, name=self.name, dtype="float64")

    @property
    def coefficients_(self) -> pd.Series:
        """The fitted coefficient on each feature column.

        Returns:
            A ``float64`` series indexed by feature name, in the column order
            the model was fitted on.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        coef = np.asarray(self._model.coef_, dtype="float64").ravel()
        return pd.Series(
            coef,
            index=pd.Index(self._columns, name="feature"),
            name="coefficient",
            dtype="float64",
        )

    @property
    def intercept_(self) -> float:
        """The fitted intercept, or ``0.0`` when ``fit_intercept`` is false.

        Returns:
            The intercept term.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        _require_fitted(self._fitted, self.name)
        return float(np.asarray(self._model.intercept_, dtype="float64").ravel()[0])
