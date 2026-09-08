"""Tests for the LSTM forecaster.

Everything here is deliberately tiny - four units, two epochs, sequences of five
rows - because what is being checked is the *wiring* (index alignment, the
direction the sequence window looks in, the chronological validation slice,
reproducibility), not whether a network can learn daily equity returns in two
epochs. Fitted models are cached at module scope so the whole file needs only a
handful of training runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_forecast.config import LstmConfig
from equity_forecast.models.base import build_model
from equity_forecast.models.lstm import LstmForecaster, chronological_validation_split
from equity_forecast.models.naive import NotFittedError

SEQ = 5
TRAIN_FRAC = 0.7
TICKERS = ("AAA", "BBB")
FEATURES = ["ret_lag_1", "ret_lag_2", "roll_mean_5", "rvol_21", "dow_mon"]


def toy_dataset(n_days: int = 60, seed: int = 5) -> pd.DataFrame:
    """Build a small schema-shaped ``(date, ticker)`` frame of features plus ``y``.

    Written locally rather than imported from ``conftest`` so this file has no
    cross-module import to resolve; the shape is what matters here, not the
    statistics.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days, name="date")
    blocks = []
    for ticker in TICKERS:
        r = pd.Series(rng.normal(0.0, 0.01, n_days), index=dates)
        block = pd.DataFrame(index=dates)
        block["ret_lag_1"] = r.shift(1)
        block["ret_lag_2"] = r.shift(2)
        block["roll_mean_5"] = r.rolling(5).mean().shift(1)
        block["rvol_21"] = r.rolling(10).std(ddof=1).shift(1)
        block["dow_mon"] = (dates.weekday == 0).astype("float64")
        block["y"] = r
        block["ticker"] = ticker
        blocks.append(block.reset_index())
    out = pd.concat(blocks, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    return out.dropna()[[*FEATURES, "y"]].astype("float64")


def tiny_config(**overrides: object) -> LstmConfig:
    """A network small enough that fitting it costs a second or two."""
    base = {
        "sequence_length": SEQ,
        "n_layers": 1,
        "units": 4,
        "dropout": 0.1,
        "learning_rate": 0.01,
        "epochs": 2,
        "batch_size": 32,
        "patience": 2,
        "val_frac": 0.2,
        "pooled": True,
        "embedding_dim": 2,
    }
    base.update(overrides)
    return LstmConfig(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """One toy dataset shared by the whole module."""
    return toy_dataset()


@pytest.fixture(scope="module")
def window(dataset: pd.DataFrame) -> pd.DataFrame:
    """The fold's full window: features for two tickers over ~60 business days."""
    return dataset[FEATURES]


@pytest.fixture(scope="module")
def targets(dataset: pd.DataFrame) -> pd.Series:
    """Targets aligned with :func:`window`."""
    return dataset["y"]


@pytest.fixture(scope="module")
def split_date(window: pd.DataFrame) -> pd.Timestamp:
    """Last date of the training block."""
    dates = pd.DatetimeIndex(window.index.get_level_values("date").unique()).sort_values()
    return dates[int(len(dates) * TRAIN_FRAC)]


@pytest.fixture(scope="module")
def train_mask(window: pd.DataFrame, split_date: pd.Timestamp) -> np.ndarray:
    """Boolean mask picking the training rows out of the window."""
    return np.asarray(window.index.get_level_values("date") <= split_date)


@pytest.fixture(scope="module")
def fitted(window: pd.DataFrame, targets: pd.Series, train_mask: np.ndarray) -> LstmForecaster:
    """A pooled model fitted once and reused by several tests."""
    model = LstmForecaster(tiny_config(), seed=0)
    return model.fit(window[train_mask], targets[train_mask])


@pytest.fixture(scope="module")
def predictions(fitted: LstmForecaster, window: pd.DataFrame) -> pd.Series:
    """Predictions over the full window from the cached pooled model."""
    return fitted.predict(window)


# --------------------------------------------------------------------------- #
# shape / index / dtype contract
# --------------------------------------------------------------------------- #


def test_predict_matches_the_window_index_exactly(
    predictions: pd.Series, window: pd.DataFrame
) -> None:
    assert isinstance(predictions, pd.Series)
    assert predictions.index.equals(window.index)
    assert len(predictions) == len(window)
    assert predictions.name == "lstm"
    assert predictions.dtype == np.dtype("float64")
    assert np.isfinite(predictions.to_numpy()).all()


def test_warmup_rows_fall_back_to_the_training_mean(
    predictions: pd.Series, window: pd.DataFrame
) -> None:
    """The first ``SEQ - 1`` rows of each ticker have no full history of their own."""
    dates = pd.DatetimeIndex(window.index.get_level_values("date").unique()).sort_values()
    warmup = dates[: SEQ - 1]
    for ticker in TICKERS:
        head = predictions.xs(ticker, level="ticker").loc[warmup]
        assert np.unique(head.to_numpy()).size == 1, "warm-up rows share one constant"
        # ...and the row right after warm-up is a real network output, not that constant.
        after = predictions.xs(ticker, level="ticker").loc[dates[SEQ - 1]]
        assert not np.isclose(after, head.iloc[0])


def test_unseen_ticker_is_predicted_without_crashing(
    fitted: LstmForecaster, window: pd.DataFrame
) -> None:
    """A name absent from training maps to the reserved embedding row 0."""
    assert set(fitted.ticker_vocabulary_) == {"AAA", "BBB"}
    assert 0 not in set(fitted.ticker_vocabulary_.values())

    extra = window.xs("AAA", level="ticker").copy()
    extra.index = pd.MultiIndex.from_product([extra.index, ["ZZZ"]], names=["date", "ticker"])
    widened = pd.concat([window, extra]).sort_index()

    out = fitted.predict(widened)
    assert out.index.equals(widened.index)
    assert np.isfinite(out.to_numpy()).all()


def test_predict_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        LstmForecaster(tiny_config(), seed=0).predict(pd.DataFrame())


# --------------------------------------------------------------------------- #
# determinism (NFR-1)
# --------------------------------------------------------------------------- #


def test_same_seed_gives_bitwise_identical_predictions(
    window: pd.DataFrame, targets: pd.Series, train_mask: np.ndarray, predictions: pd.Series
) -> None:
    twin = LstmForecaster(tiny_config(), seed=0).fit(window[train_mask], targets[train_mask])
    again = twin.predict(window)
    np.testing.assert_array_equal(again.to_numpy(), predictions.to_numpy())


def test_a_different_seed_gives_different_predictions(
    window: pd.DataFrame, targets: pd.Series, train_mask: np.ndarray, predictions: pd.Series
) -> None:
    other = LstmForecaster(tiny_config(), seed=12345).fit(window[train_mask], targets[train_mask])
    out = other.predict(window)
    assert not np.allclose(out.to_numpy(), predictions.to_numpy())


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #


def test_perturbing_the_future_leaves_earlier_predictions_untouched(
    fitted: LstmForecaster, window: pd.DataFrame, predictions: pd.Series
) -> None:
    """The forecast for row ``t`` must be a function of rows ``<= t`` only."""
    dates = pd.DatetimeIndex(window.index.get_level_values("date").unique()).sort_values()
    cut = dates[len(dates) // 2]

    tampered = window.copy()
    later = np.asarray(tampered.index.get_level_values("date") > cut)
    tampered.loc[later, :] = tampered.loc[later, :] + 7.5

    out = fitted.predict(tampered)
    upto = np.asarray(window.index.get_level_values("date") <= cut)
    np.testing.assert_array_equal(out.to_numpy()[upto], predictions.to_numpy()[upto])
    # The perturbation must actually reach *something*, or the test proves nothing.
    assert not np.allclose(out.to_numpy()[~upto], predictions.to_numpy()[~upto])


# --------------------------------------------------------------------------- #
# the chronological validation split (tested without training anything)
# --------------------------------------------------------------------------- #


def test_validation_split_is_the_last_fraction_of_dates_in_time_order() -> None:
    dates = pd.bdate_range("2020-01-01", periods=100, name="date")
    train, val = chronological_validation_split(dates, 0.15)

    assert len(val) == 15
    assert len(train) == 85
    # The boundary itself: the split is a single cut in time, not a sample.
    assert train.max() < val.min()
    assert list(val) == list(dates[-15:])
    assert list(train) == list(dates[:85])
    assert train.append(val).equals(dates)


def test_validation_split_handles_duplicates_disorder_and_edges() -> None:
    dates = pd.bdate_range("2021-01-04", periods=10)
    shuffled = pd.DatetimeIndex(list(dates[::-1]) + list(dates))  # duplicated and reversed

    train, val = chronological_validation_split(shuffled, 0.2)
    assert list(val) == list(dates[-2:])
    assert train.max() < val.min()

    assert len(chronological_validation_split(dates, 0.0)[1]) == 0
    # A degenerate fraction still leaves something to train on.
    assert len(chronological_validation_split(dates, 1.0)[0]) == 1
    with pytest.raises(ValueError, match="val_frac"):
        chronological_validation_split(dates, 1.5)


def test_fitted_validation_slice_never_touches_the_test_block(
    fitted: LstmForecaster, window: pd.DataFrame, split_date: pd.Timestamp
) -> None:
    """Early stopping saw only the tail of the *training* fold."""
    train_dates = pd.DatetimeIndex(
        window.index.get_level_values("date")[
            window.index.get_level_values("date") <= split_date
        ].unique()
    ).sort_values()
    val = fitted.validation_dates_

    assert len(val) == int(len(train_dates) * 0.2)
    assert val.max() == train_dates.max()
    assert set(val).issubset(set(train_dates))
    assert val.min() > train_dates[: len(train_dates) - len(val)].max()


# --------------------------------------------------------------------------- #
# per-ticker mode and the short-history fallback
# --------------------------------------------------------------------------- #


def test_unpooled_fits_one_network_per_ticker_and_still_covers_the_index(
    window: pd.DataFrame, targets: pd.Series, train_mask: np.ndarray
) -> None:
    model = LstmForecaster(tiny_config(pooled=False, dropout=0.0), seed=3)
    model.fit(window[train_mask], targets[train_mask])

    assert model.n_networks_ == 2

    out = model.predict(window)
    assert out.index.equals(window.index)
    assert np.isfinite(out.to_numpy()).all()
    assert out.name == "lstm"
    # Independent networks: the two tickers do not produce the same series.
    assert not np.allclose(
        out.xs("AAA", level="ticker").to_numpy(), out.xs("BBB", level="ticker").to_numpy()
    )


def test_training_fold_shorter_than_the_sequence_falls_back_to_the_mean(
    window: pd.DataFrame, targets: pd.Series, caplog: pytest.LogCaptureFixture
) -> None:
    short = window.groupby(level="ticker").head(SEQ - 2)
    short_y = targets.loc[short.index]

    with caplog.at_level("WARNING", logger="equity_forecast.models.lstm"):
        model = LstmForecaster(tiny_config(), seed=0).fit(short, short_y)

    assert "falling back to the training mean" in caplog.text
    assert model.n_networks_ == 0

    out = model.predict(window)
    assert out.index.equals(window.index)
    assert np.isfinite(out.to_numpy()).all()
    # One constant per ticker: the per-ticker training mean, and nothing else.
    for ticker in TICKERS:
        column = out.xs(ticker, level="ticker")
        assert np.unique(column.to_numpy()).size == 1
        assert np.isclose(column.iloc[0], short_y.xs(ticker, level="ticker").mean())


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_registry_lookup_builds_the_forecaster() -> None:
    model = build_model("lstm", tiny_config(), seed=0)
    assert isinstance(model, LstmForecaster)
    assert model.name == "lstm"
    assert model.predict_vol(pd.DataFrame()) is None


# --------------------------------------------------------------------------- #
# Regression: the network's output must live on the target's scale.
#
# The head used to be Glorot-initialised while the target is a daily log return
# with a standard deviation near 0.01. An untrained network therefore emitted
# values ~100x too large, and a short training run never walked that offset
# back: on the smoke run the LSTM's forecasts had std 0.0594 against a target
# std of 0.0097 and a maximum of +0.52, making its RMSE six times the
# random-walk baseline's. These tests pin both halves of the fix.
# --------------------------------------------------------------------------- #
def test_untrained_network_predicts_no_edge(window: pd.DataFrame, targets: pd.Series) -> None:
    """With zero training epochs the model must fall back to ~the training mean.

    The head is zero-initialised precisely so that a network which has learned
    nothing predicts the random-walk baseline rather than noise. If this fails,
    the model degrades to garbage instead of to the benchmark.
    """
    train = window.index.get_level_values("date") <= window.index.get_level_values("date")[-40]
    model = LstmForecaster(tiny_config(epochs=1, learning_rate=0.0), seed=3).fit(
        window[train], targets[train]
    )
    predictions = model.predict(window)

    assert np.isfinite(predictions).all()
    # Every prediction sits at the training mean, i.e. no edge is claimed.
    np.testing.assert_allclose(predictions.to_numpy(), float(targets[train].mean()), atol=5e-4)

    # The standardisation constants come from the TRAIN fold only - using the
    # full window's mean and standard deviation would be look-ahead.
    train_targets = targets[train]
    assert model._target_mean == pytest.approx(float(train_targets.mean()))
    assert model._target_std == pytest.approx(float(train_targets.std(ddof=0)))
    assert model._target_mean != pytest.approx(float(targets.mean()))


def test_predictions_are_on_the_scale_of_the_target(
    window: pd.DataFrame, targets: pd.Series
) -> None:
    """Forecast dispersion must not dwarf the quantity being forecast.

    MSE training shrinks predictions TOWARD the conditional mean, so a forecast
    series with many times the target's dispersion is a broken model, not a
    confident one. Three times the target's standard deviation is already far
    more spread than any honest daily-return forecast; the pre-fix model was at
    six times and would fail this.
    """
    train = window.index.get_level_values("date") <= window.index.get_level_values("date")[-40]
    model = LstmForecaster(tiny_config(epochs=3), seed=5).fit(window[train], targets[train])
    predictions = model.predict(window)

    target_std = float(targets.std(ddof=1))
    assert float(predictions.std(ddof=1)) < 3.0 * target_std
    assert float(predictions.abs().max()) < 10.0 * target_std
