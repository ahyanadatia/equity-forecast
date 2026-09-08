"""equity-forecast: an honest comparison of forecasting models for equity returns.

The package predicts one-day-ahead log returns (never price levels), evaluates
every model against a random-walk baseline under walk-forward validation, tests
significance with Diebold-Mariano, and prices the result through a long/short
backtest with transaction costs.
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
