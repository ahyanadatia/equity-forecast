"""The forecaster interface every model implements.

Contract
--------
A forecaster is fitted on the *training* rows of a walk-forward fold and then
asked to predict over the fold's **full window** (train rows, embargo gap rows
and test rows), returning a series indexed exactly like the frame it was given.
The evaluator slices out the test rows itself.

Passing the whole window - rather than the test block alone - exists so that
sequence models (LSTM) and state-space models (ARIMA, GARCH) can warm up their
state on training history instead of starting cold on the first test day. It
places one obligation on implementations, which
:func:`tests.test_models.test_predictions_are_causal` enforces for every
registered model:

    **The prediction for row ``t`` must not depend on any row after ``t``.**

Feature rows are already constructed so that row ``t`` contains only information
observable at the close of ``t-1`` (see :mod:`equity_forecast.features`), so a
causal model reading row ``t`` is using no future information.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, runtime_checkable

import pandas as pd

__all__ = [
    "MODEL_REGISTRY",
    "BaseForecaster",
    "Forecaster",
    "ModelParams",
    "available_models",
    "build_model",
    "register_model",
    "returns_from_features",
]

# Each model is configured by its own frozen dataclass from `config.py` (or a plain dict for
# the parameterless baselines). There is no useful common supertype, and the registry has to
# hold all of them, so the parameter bundle is deliberately dynamic at this boundary.
ModelParams: TypeAlias = Any


@runtime_checkable
class Forecaster(Protocol):
    """Structural type for a point forecaster of returns."""

    name: str

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Forecaster:
        """Fit on the training rows of one fold and return ``self``."""
        ...

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Return a point forecast of the target, indexed exactly like ``x``."""
        ...

    def predict_vol(self, x: pd.DataFrame) -> pd.Series | None:
        """Return a conditional volatility forecast, or ``None`` if unsupported."""
        ...


class BaseForecaster(abc.ABC):
    """Convenience base class: implements ``predict_vol`` as "not supported"."""

    name: str = "base"

    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> BaseForecaster:
        """Fit the model on ``(x, y)`` and return ``self``."""

    @abc.abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict the target for every row of ``x``."""

    def predict_vol(self, x: pd.DataFrame) -> pd.Series | None:
        """Return ``None``: most models forecast the mean, not the variance.

        Args:
            x: Feature frame; ignored by the default implementation.

        Returns:
            ``None``.
        """
        del x
        return None

    def __repr__(self) -> str:
        """Return a short, stable representation."""
        return f"{type(self).__name__}(name={self.name!r})"


MODEL_REGISTRY: dict[str, type[BaseForecaster]] = {}


def register_model(name: str) -> Callable[[type[BaseForecaster]], type[BaseForecaster]]:
    """Class decorator registering a forecaster under ``name``.

    Args:
        name: Registry key, matching the model's key in ``models:`` config.

    Returns:
        The decorator.
    """

    def decorator(cls: type[BaseForecaster]) -> type[BaseForecaster]:
        if name in MODEL_REGISTRY:
            msg = f"model {name!r} is already registered"
            raise ValueError(msg)
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def available_models() -> list[str]:
    """Return the sorted names of all registered models."""
    return sorted(MODEL_REGISTRY)


def build_model(name: str, params: ModelParams, seed: int) -> BaseForecaster:
    """Construct a registered forecaster.

    Args:
        name: Registry key, e.g. ``"ridge"``.
        params: The model's config dataclass (or ``dict`` for parameterless models).
        seed: Random seed for models that need one.

    Returns:
        A freshly constructed, unfitted forecaster.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in MODEL_REGISTRY:
        msg = f"unknown model {name!r}; available: {available_models()}"
        raise KeyError(msg)
    return MODEL_REGISTRY[name](params, seed)  # type: ignore[call-arg]


def returns_from_features(x: pd.DataFrame) -> pd.DataFrame:
    """Recover the daily log-return panel implied by a feature frame.

    Row ``t`` of the feature frame carries ``ret_lag_1``, the realised return of
    day ``t-1``. Shifting that column back by one position within each ticker
    therefore recovers the return of day ``t`` itself - using only information
    that was observable by the close of ``t``, never later. The final row of each
    ticker becomes ``NaN``: its return is genuinely unknown when the forecast for
    that day has to be made, which is exactly the point.

    Args:
        x: Feature frame indexed by ``(date, ticker)`` containing ``ret_lag_1``.

    Returns:
        A wide frame indexed by date with one column per ticker, holding the
        realised log return of each day, ``NaN`` on each ticker's last row.

    Raises:
        KeyError: If ``x`` has no ``ret_lag_1`` column.
    """
    if "ret_lag_1" not in x.columns:
        msg = "returns_from_features requires a 'ret_lag_1' column"
        raise KeyError(msg)
    shifted = x["ret_lag_1"].groupby(level="ticker").shift(-1)
    # `unstack` is exactly right here: one row per date, one column per ticker, no aggregation.
    return shifted.unstack("ticker").sort_index()  # noqa: PD010
