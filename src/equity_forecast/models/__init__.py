"""Forecasting models.

Importing this package imports every model module, which is what populates
:data:`equity_forecast.models.base.MODEL_REGISTRY`. Anything that resolves a
model by name - the pipeline, the CLI, the cross-model test suite - should
import from here rather than from the individual modules, so the registry is
guaranteed to be complete.
"""

from __future__ import annotations

from equity_forecast.models.arima import ArimaForecaster
from equity_forecast.models.base import (
    MODEL_REGISTRY,
    BaseForecaster,
    Forecaster,
    ModelParams,
    available_models,
    build_model,
    register_model,
    returns_from_features,
)
from equity_forecast.models.garch import ArimaGarchForecaster
from equity_forecast.models.linear import Ridge
from equity_forecast.models.lstm import LstmForecaster
from equity_forecast.models.naive import HistoricalMean, ZeroReturn

__all__ = [
    "MODEL_REGISTRY",
    "ArimaForecaster",
    "ArimaGarchForecaster",
    "BaseForecaster",
    "Forecaster",
    "HistoricalMean",
    "LstmForecaster",
    "ModelParams",
    "Ridge",
    "ZeroReturn",
    "available_models",
    "build_model",
    "register_model",
    "returns_from_features",
]
