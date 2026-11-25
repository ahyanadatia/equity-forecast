"""Shared pytest fixtures.

Everything here is offline: the price fixtures are the committed synthetic
parquets in ``tests/fixtures/prices``, and ``synthetic_dataset`` hand-builds a
frame conforming to the documented dataset schema without importing
:mod:`equity_forecast.features`, so model and backtest tests stay independent of
the feature pipeline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
"""Whether the optional ``lstm`` extra is installed."""


def _tensorflow_importable() -> bool:
    """Return True only if TensorFlow actually imports, not just exists on the path."""
    if not HAS_TENSORFLOW:
        return False
    try:
        import tensorflow  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip LSTM tests cleanly when TensorFlow is not installed.

    A test needs TensorFlow if it is marked ``requires_tf``, lives in
    ``test_lstm.py``, or is a ``[lstm]`` parametrisation of a model contract test.
    """
    del config
    needs_tf = [
        item
        for item in items
        if item.get_closest_marker("requires_tf") is not None
        or item.path.name == "test_lstm.py"
        or "[lstm]" in item.nodeid
    ]
    if not needs_tf or _tensorflow_importable():
        return
    skip = pytest.mark.skip(reason="TensorFlow not installed (pip install -e '.[lstm]')")
    for item in needs_tf:
        item.add_marker(skip)


FIXTURE_PRICES = Path(__file__).resolve().parent / "fixtures" / "prices"

TOY_FEATURE_COLUMNS: list[str] = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "roll_mean_5",
    "roll_std_5",
    "max_dd_21",
    "rvol_21",
    "dow_mon",
    "dow_tue",
    "dow_wed",
    "dow_thu",
    "dow_fri",
]
TOY_CONTINUOUS_COLUMNS: list[str] = [c for c in TOY_FEATURE_COLUMNS if not c.startswith("dow_")]


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Return the directory holding the committed price parquets."""
    return FIXTURE_PRICES


@pytest.fixture(scope="session")
def raw_fixture_frames() -> dict[str, pd.DataFrame]:
    """Load every fixture parquet keyed by ticker, straight from disk."""
    return {p.stem: pd.read_parquet(p) for p in sorted(FIXTURE_PRICES.glob("*.parquet"))}


@pytest.fixture(scope="session")
def price_panel(raw_fixture_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a wide close-price panel (date x ticker) for SYNA and SYNB."""
    panel = pd.DataFrame({t: raw_fixture_frames[t]["close"] for t in ("SYNA", "SYNB")})
    panel.index.name = "date"
    panel.columns.name = "ticker"
    return panel.sort_index()


def make_synthetic_dataset(
    n_days: int = 400,
    tickers: tuple[str, ...] = ("AAA", "BBB"),
    seed: int = 7,
    signal_strength: float = 0.0,
) -> pd.DataFrame:
    """Build a dataset frame conforming to the documented schema.

    The frame is internally consistent: ``ret_lag_k`` at row ``t`` really is the
    return of row ``t-k``, and ``y == ret_1d`` (horizon 1), so models that
    reconstruct a return panel from lagged features get the right answer.

    Args:
        n_days: Number of business days to generate.
        tickers: Ticker names to generate.
        seed: Seed for the generator.
        signal_strength: Coefficient linking ``ret_lag_1`` to the next return.
            ``0.0`` gives an unpredictable series; a positive value makes the
            series genuinely forecastable, which some tests rely on.

    Returns:
        A frame with a ``(date, ticker)`` MultiIndex, the toy feature columns,
        ``y`` and ``ret_1d``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_days, name="date")
    blocks: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        noise = rng.normal(0.0, 0.012, n_days + 70)
        r = np.empty_like(noise)
        r[0] = noise[0]
        for t in range(1, len(noise)):
            r[t] = signal_strength * r[t - 1] + noise[t]
        series = pd.Series(r[70:], index=dates)

        block = pd.DataFrame(index=dates)
        for k in (1, 2, 3):
            block[f"ret_lag_{k}"] = series.shift(k)
        block["roll_mean_5"] = series.rolling(5).mean().shift(1)
        block["roll_std_5"] = series.rolling(5).std(ddof=1).shift(1)
        cum = series.cumsum()
        running_max = cum.rolling(21).max()
        block["max_dd_21"] = (running_max - cum).rolling(21).max().shift(1).abs()
        block["rvol_21"] = np.sqrt(252.0 * series.pow(2).rolling(21).mean()).shift(1)
        weekday = dates.weekday
        for j, day in enumerate(("mon", "tue", "wed", "thu", "fri")):
            block[f"dow_{day}"] = (weekday == j).astype("float64")
        block["y"] = series
        block["ret_1d"] = series
        block["ticker"] = ticker
        block.index.name = "date"
        blocks.append(block.reset_index().assign(_order=i))

    out = pd.concat(blocks, ignore_index=True)
    out = out.set_index(["date", "ticker"]).drop(columns="_order").sort_index()
    return out.dropna()[[*TOY_FEATURE_COLUMNS, "y", "ret_1d"]].astype("float64")


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    """A schema-conforming, unpredictable toy dataset."""
    return make_synthetic_dataset()


@pytest.fixture
def predictable_dataset() -> pd.DataFrame:
    """A schema-conforming toy dataset with a genuine AR(1) signal to find."""
    return make_synthetic_dataset(signal_strength=0.35, seed=11)
