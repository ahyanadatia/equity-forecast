"""The cross-cutting contract every registered forecaster has to satisfy.

``models/base.py`` states the interface in prose; this module turns it into
assertions and runs them against **every** entry of
:data:`~equity_forecast.models.base.MODEL_REGISTRY`, so a new model cannot join
the comparison without honouring the same rules as the ones already in it.

The headline test is :func:`test_predictions_are_causal` (NFR-2), which
``models/base.py`` names directly. Every model is handed the fold's *entire*
window - training rows, embargo gap and test rows - so that sequence and
state-space models can warm up. That convenience is exactly what makes
look-ahead possible: a model that filters the whole window at once and reads its
smoothed (rather than filtered) state would score beautifully in backtest and be
worthless in production. The test forecloses it empirically. It fits once, keeps
the fitted model, then re-predicts on copies of the window whose feature values
*after* a cut date have been mangled - scaled up threefold, then replaced with
noise - and requires the predictions on or before the cut to be bit-for-bit
unmoved. A model that peeks cannot pass.

Everything here is offline and deliberately small: the models are built from
cut-down configs (a 4-unit LSTM trained for a handful of epochs, ARIMA(1,0,0))
because these tests check interface behaviour, not forecasting skill. The one
test that *does* care about skill is
:func:`test_ridge_finds_the_ar1_signal_but_zero_return_does_not`: it confirms
the harness can detect a signal that genuinely exists, so that a null result on
real market data can be read as a fact about markets rather than a broken
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandas.testing import assert_index_equal, assert_series_equal

from equity_forecast.config import ArimaConfig, ArimaGarchConfig, LstmConfig, RidgeConfig
from equity_forecast.metrics import directional_accuracy, rmse

# Importing the model modules is what executes their ``@register_model``
# decorators, and therefore what populates the registry. They are named
# explicitly rather than relied upon transitively so that this suite keeps
# covering every model even if the package ``__init__`` is reorganised.
from equity_forecast.models import arima, garch, linear, lstm, naive  # noqa: F401
from equity_forecast.models.base import (
    MODEL_REGISTRY,
    BaseForecaster,
    available_models,
    build_model,
)

SEED = 42

#: Every key the project promises to ship, per the module contract.
EXPECTED_MODELS: tuple[str, ...] = (
    "arima",
    "arima_garch",
    "historical_mean",
    "lstm",
    "ridge",
    "zero_return",
)

#: Deliberately tiny hyperparameters. These tests assert on interface behaviour,
#: so the only thing that matters is that every model fits in about a second.
TINY_PARAMS: dict[str, Any] = {
    "zero_return": {},
    "historical_mean": {},
    "arima": ArimaConfig(order=(1, 0, 0), auto=False),
    "arima_garch": ArimaGarchConfig(mean_lags=1, p=1, q=1, dist="normal"),
    "ridge": RidgeConfig(alpha=1.0),
    "lstm": LstmConfig(
        sequence_length=5,
        n_layers=1,
        units=4,
        dropout=0.0,
        learning_rate=0.05,
        epochs=8,
        batch_size=64,
        patience=3,
        val_frac=0.2,
        pooled=True,
        embedding_dim=2,
    ),
}

MODEL_NAMES: list[str] = sorted(MODEL_REGISTRY)

#: Positions in the window calendar at which the causality probe cuts. Both sit
#: inside the test block, past ``train_frac``, so the perturbation lands on rows
#: the model has never been fitted on.
CUT_FRACTIONS: tuple[float, ...] = (0.70, 0.85)

#: How a perturbation mangles the future rows of a window.
PERTURBATIONS: tuple[str, ...] = ("tripled", "noise")


# --------------------------------------------------------------------------- #
# window construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    """One walk-forward window: scaled features, targets and the train mask.

    Attributes:
        x: Scaled feature frame for the whole window (train + test rows), which
            is what ``predict`` is contractually handed.
        y: Targets for the whole window.
        train_mask: Boolean row mask selecting the training block.
        row_dates: The date of every row of ``x``, in row order.
        dates: The window's unique dates, sorted.
    """

    x: pd.DataFrame
    y: pd.Series
    train_mask: npt.NDArray[np.bool_]
    row_dates: pd.DatetimeIndex
    dates: pd.DatetimeIndex

    @property
    def train_x(self) -> pd.DataFrame:
        """Return the training rows of the feature frame."""
        return self.x[self.train_mask]

    @property
    def train_y(self) -> pd.Series:
        """Return the training targets."""
        return self.y[self.train_mask]

    @property
    def test_mask(self) -> npt.NDArray[np.bool_]:
        """Return the boolean row mask selecting the out-of-sample block."""
        return ~self.train_mask

    def cut_at(self, fraction: float) -> pd.Timestamp:
        """Return the window date sitting ``fraction`` of the way through it."""
        return self.dates[int(len(self.dates) * fraction)]


def standardise(x_raw: pd.DataFrame, train_mask: npt.NDArray[np.bool_]) -> pd.DataFrame:
    """Z-score the continuous columns using training-fold statistics only.

    The scaling is done here rather than by importing
    :mod:`equity_forecast.features` so that a bug in the feature pipeline cannot
    make the model contract suite fail (or, worse, pass).

    Args:
        x_raw: Unscaled feature frame.
        train_mask: Boolean row mask selecting the training block.

    Returns:
        A copy of ``x_raw`` with every non ``dow_`` column standardised.
    """
    continuous = [c for c in x_raw.columns if not str(c).startswith("dow_")]
    train = x_raw.loc[train_mask, continuous]
    mean = train.mean()
    std = train.std(ddof=0).replace(0.0, 1.0)
    scaled = x_raw.copy()
    scaled[continuous] = (x_raw[continuous] - mean) / std
    return scaled.astype("float64")


def split_window(dataset: pd.DataFrame, train_frac: float = 0.6) -> Window:
    """Split a dataset chronologically into a train block and a full window.

    Args:
        dataset: A schema-conforming dataset frame (features, ``y``, ``ret_1d``).
        train_frac: Fraction of the calendar used for training.

    Returns:
        The resulting :class:`Window`.
    """
    row_dates = pd.DatetimeIndex(dataset.index.get_level_values("date"))
    dates = pd.DatetimeIndex(row_dates.unique()).sort_values()
    train_end = dates[int(len(dates) * train_frac)]
    train_mask = np.asarray(row_dates <= train_end, dtype=bool)
    x_raw = dataset.drop(columns=["y", "ret_1d"])
    return Window(
        x=standardise(x_raw, train_mask),
        y=dataset["y"].astype("float64"),
        train_mask=train_mask,
        row_dates=row_dates,
        dates=dates,
    )


@pytest.fixture
def window(synthetic_dataset: pd.DataFrame) -> Window:
    """A train/window split of the unpredictable toy dataset."""
    return split_window(synthetic_dataset)


# --------------------------------------------------------------------------- #
# fitted-model cache
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FittedModel:
    """A model fitted once and shared by the tests that only read from it.

    Attributes:
        model: The object returned by ``fit``.
        fit_returned_self: Whether ``fit`` returned the instance it was called
            on, as the contract requires.
        train_index: The index ``fit`` was called with, used to detect a caller
            asking for a model fitted on different data.
    """

    model: BaseForecaster
    fit_returned_self: bool
    train_index: pd.Index


_FITTED: dict[str, FittedModel] = {}


def fitted(name: str, window: Window) -> FittedModel:
    """Fit ``name`` on ``window``'s training block, reusing an earlier fit.

    The toy datasets are generated from fixed seeds, so every test sees the same
    training rows; caching keeps the LSTM to a single training run instead of
    one per test. The cached entry records the training index and is discarded
    if a caller ever arrives with different data.

    Args:
        name: Registry key of the model to fit.
        window: The window whose training block to fit on.

    Returns:
        The cached :class:`FittedModel`.
    """
    cached = _FITTED.get(name)
    if cached is not None and cached.train_index.equals(window.train_x.index):
        return cached
    model = build_model(name, TINY_PARAMS[name], SEED)
    returned = model.fit(window.train_x, window.train_y)
    record = FittedModel(
        model=returned,
        fit_returned_self=returned is model,
        train_index=window.train_x.index,
    )
    _FITTED[name] = record
    return record


def perturb_future(x: pd.DataFrame, after: npt.NDArray[np.bool_], kind: str) -> pd.DataFrame:
    """Return a copy of ``x`` whose rows selected by ``after`` are mangled.

    Args:
        x: The feature frame to copy.
        after: Boolean row mask selecting the rows to overwrite.
        kind: ``"tripled"`` multiplies those rows by three; ``"noise"`` replaces
            them with draws from a fixed-seed normal.

    Returns:
        A new frame with the same index, columns and dtypes as ``x``.

    Raises:
        ValueError: If ``kind`` is not a known perturbation.
    """
    values = np.asarray(x.to_numpy(dtype="float64"), dtype="float64").copy()
    if kind == "tripled":
        values[after] *= 3.0
    elif kind == "noise":
        rng = np.random.default_rng(1234)
        values[after] = rng.normal(0.0, 1.5, size=values[after].shape)
    else:  # pragma: no cover - guards a typo in the parametrisation
        msg = f"unknown perturbation {kind!r}"
        raise ValueError(msg)
    return pd.DataFrame(values, index=x.index, columns=x.columns)


def with_unseen_ticker(x: pd.DataFrame, source: str, unseen: str) -> pd.DataFrame:
    """Return ``x`` with an extra ticker the model has never been fitted on.

    Args:
        x: Feature frame indexed by ``(date, ticker)``.
        source: Ticker whose rows are copied to create the new one.
        unseen: Name of the new ticker.

    Returns:
        A sorted frame carrying every original row plus the new ticker's.
    """
    block = x.xs(source, level="ticker", drop_level=False) * 1.1
    block.index = pd.MultiIndex.from_arrays(
        [
            block.index.get_level_values("date"),
            pd.Index([unseen] * len(block), name="ticker"),
        ],
        names=["date", "ticker"],
    )
    return pd.concat([x, block]).sort_index()


# --------------------------------------------------------------------------- #
# 1. the predict() interface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predict_returns_a_series_indexed_exactly_like_its_input(name: str, window: Window) -> None:
    record = fitted(name, window)
    assert record.fit_returned_self, f"{name}.fit() must return self"

    pred = record.model.predict(window.x)

    assert isinstance(pred, pd.Series)
    assert_index_equal(pred.index, window.x.index)
    assert pred.dtype == np.dtype("float64")
    assert pred.name == name
    values = pred.to_numpy()
    assert not np.isnan(values).any(), f"{name} produced NaN forecasts"
    assert np.isfinite(values).all(), f"{name} produced non-finite forecasts"


# --------------------------------------------------------------------------- #
# 2. NFR-2: no look-ahead
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_causal(name: str, window: Window) -> None:
    """Mangling the future must not move a single forecast at or before the cut."""
    record = fitted(name, window)
    baseline = record.model.predict(window.x)

    for fraction in CUT_FRACTIONS:
        cut = window.cut_at(fraction)
        after = np.asarray(window.row_dates > cut, dtype=bool)
        assert after.any(), "the cut must leave some future rows to perturb"
        assert (~after).any(), "the cut must leave some past rows to check"

        for kind in PERTURBATIONS:
            mangled = perturb_future(window.x, after, kind)
            assert not mangled.equals(window.x), "the perturbation changed nothing"

            pred = record.model.predict(mangled)
            assert_index_equal(pred.index, window.x.index)

            before = np.abs(pred.to_numpy()[~after] - baseline.to_numpy()[~after]).max()
            assert before <= 1e-10, (
                f"{name} looks ahead: {kind} perturbation after {cut.date()} moved a "
                f"forecast at or before the cut by {before:.3e}"
            )


def test_the_causality_probe_actually_moves_predictions(window: Window) -> None:
    """Guard against a vacuous causality test: the probe must have teeth.

    ``zero_return`` and ``historical_mean`` ignore the features entirely, so they
    pass :func:`test_predictions_are_causal` trivially. The models that *do* read
    the feature frame must visibly react to a mangled future - otherwise the
    perturbation is not reaching them and the causality test proves nothing.
    """
    cut = window.cut_at(CUT_FRACTIONS[0])
    after = np.asarray(window.row_dates > cut, dtype=bool)
    mangled = perturb_future(window.x, after, "tripled")

    for name in ("ridge", "arima", "arima_garch"):
        model = fitted(name, window).model
        moved = np.abs(
            model.predict(mangled).to_numpy()[after] - model.predict(window.x).to_numpy()[after]
        ).max()
        assert moved > 1e-6, f"the perturbation never reached {name}"


# --------------------------------------------------------------------------- #
# 3. forecasts have to be plausible daily log returns
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_have_a_sane_magnitude(name: str, window: Window) -> None:
    """A daily log-return forecast of 50% is a bug, not a signal."""
    pred = fitted(name, window).model.predict(window.x)
    worst = float(np.abs(pred.to_numpy()).max())
    assert worst < 0.5, f"{name} forecast a daily log return of {worst:.3f}"


# --------------------------------------------------------------------------- #
# 4. the optional volatility forecast
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predict_vol_is_none_or_a_positive_series(name: str, window: Window) -> None:
    vol = fitted(name, window).model.predict_vol(window.x)
    if vol is None:
        return
    assert isinstance(vol, pd.Series)
    assert_index_equal(vol.index, window.x.index)
    assert vol.dtype == np.dtype("float64")
    values = vol.to_numpy()
    assert np.isfinite(values).all(), f"{name}.predict_vol produced non-finite values"
    assert (values > 0.0).all(), f"{name}.predict_vol produced a non-positive volatility"


def test_at_least_one_model_forecasts_volatility(window: Window) -> None:
    """Vol targeting needs a real vol forecast from somewhere."""
    vol = fitted("arima_garch", window).model.predict_vol(window.x)
    assert vol is not None, "arima_garch must expose a conditional volatility forecast"


# --------------------------------------------------------------------------- #
# 5. the registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_model_constructs_every_registered_model(name: str) -> None:
    model = build_model(name, TINY_PARAMS[name], SEED)
    assert isinstance(model, BaseForecaster)
    assert model.name == name
    assert type(model) is MODEL_REGISTRY[name]


def test_available_models_lists_the_six_expected_keys() -> None:
    assert set(EXPECTED_MODELS) <= set(available_models())
    assert sorted(EXPECTED_MODELS) == sorted(MODEL_REGISTRY)


def test_every_registered_model_has_test_parameters() -> None:
    """A model added to the registry must be added to this suite too."""
    assert set(MODEL_NAMES) == set(TINY_PARAMS), (
        "TINY_PARAMS must cover exactly the registered models"
    )


# --------------------------------------------------------------------------- #
# 6. reproducibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_same_seed_and_data_give_identical_predictions(name: str, window: Window) -> None:
    first = build_model(name, TINY_PARAMS[name], SEED).fit(window.train_x, window.train_y)
    second = build_model(name, TINY_PARAMS[name], SEED).fit(window.train_x, window.train_y)
    assert_series_equal(first.predict(window.x), second.predict(window.x))


# --------------------------------------------------------------------------- #
# 7. a ticker the model has never met
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_an_unseen_ticker_in_the_window_does_not_crash(name: str, window: Window) -> None:
    """Predicting for a name absent from training may fall back, but must work."""
    model = fitted(name, window).model
    widened = with_unseen_ticker(window.x, source="BBB", unseen="CCC")
    assert "CCC" in set(widened.index.get_level_values("ticker"))

    pred = model.predict(widened)

    assert_index_equal(pred.index, widened.index)
    assert pred.name == name
    assert np.isfinite(pred.to_numpy()).all(), f"{name} produced non-finite forecasts for CCC"
    unseen = pred.xs("CCC", level="ticker")
    assert len(unseen) == len(window.dates)


# --------------------------------------------------------------------------- #
# the harness can find a signal that is really there
# --------------------------------------------------------------------------- #
def test_ridge_finds_the_ar1_signal_but_zero_return_does_not(
    predictable_dataset: pd.DataFrame,
) -> None:
    """Prove the evaluation harness detects signal when signal exists.

    ``predictable_dataset`` is an AR(1) whose next return really is a function of
    the last one, so a linear model reading ``ret_lag_1`` should beat a coin
    flip out of sample. The flat zero forecast cannot: it has no direction at
    all, which is precisely why it is the baseline.

    Without this test a null result on real market data would be ambiguous -
    genuinely unforecastable returns, or a broken pipeline. With it, the null is
    a finding about markets.
    """
    window = split_window(predictable_dataset, train_frac=0.7)
    test = window.test_mask
    truth = window.y[test]

    ridge = build_model("ridge", RidgeConfig(alpha=1.0), SEED)
    ridge.fit(window.train_x, window.train_y)
    ridge_pred = ridge.predict(window.x)[test]

    zero = build_model("zero_return", {}, SEED)
    zero.fit(window.train_x, window.train_y)
    zero_pred = zero.predict(window.x)[test]

    ridge_da = directional_accuracy(truth, ridge_pred)
    zero_da = directional_accuracy(truth, zero_pred)

    assert ridge_da > 0.55, f"ridge failed to find a real AR(1) signal (DA={ridge_da:.3f})"
    assert not zero_da > 0.55, f"the zero baseline claimed direction it cannot have ({zero_da:.3f})"
    assert ridge_da > zero_da
    assert rmse(truth, ridge_pred) < rmse(truth, zero_pred)
