"""Tests for the ARIMA and ARIMA-GARCH forecasters.

Every series here is hand-built and tiny, so the whole file runs in a couple of
seconds. The two tests that matter most are the causality tests: they perturb
feature rows strictly after a cut-off date and assert that no prediction on or
before that date moves, which is what would catch a mistake in the
``apply(refit=False)`` splicing that quietly let the future into the past.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima.model import ARIMA

from conftest import TOY_CONTINUOUS_COLUMNS, TOY_FEATURE_COLUMNS, make_synthetic_dataset
from equity_forecast.config import ArimaConfig, ArimaGarchConfig
from equity_forecast.models.arima import ArimaForecaster
from equity_forecast.models.base import build_model
from equity_forecast.models.garch import ArimaGarchForecaster
from equity_forecast.models.naive import NotFittedError

START = "2016-01-04"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def dataset_from_returns(returns: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Build a schema-conforming dataset from explicit per-ticker return paths.

    Mirrors ``conftest.make_synthetic_dataset`` but takes the returns instead of
    generating them, so a test can hand-build a constant series or a series with
    a deliberate volatility burst. ``ret_lag_k`` at row ``t`` really is the
    return of row ``t-k`` and ``y == ret_1d``, which is what the models rely on
    when they reconstruct the return panel.

    Args:
        returns: One return path per ticker; all paths must be the same length.

    Returns:
        A frame with a ``(date, ticker)`` MultiIndex, the toy feature columns,
        ``y`` and ``ret_1d``, with the warm-up rows dropped.
    """
    blocks: list[pd.DataFrame] = []
    for ticker, path in returns.items():
        r = np.asarray(path, dtype="float64")
        dates = pd.bdate_range(START, periods=len(r), name="date")
        series = pd.Series(r, index=dates)
        block = pd.DataFrame(index=dates)
        for k in (1, 2, 3):
            block[f"ret_lag_{k}"] = series.shift(k)
        block["roll_mean_5"] = series.rolling(5).mean().shift(1)
        block["roll_std_5"] = series.rolling(5).std(ddof=1).shift(1)
        cum = series.cumsum()
        running_max = cum.rolling(21).max()
        block["max_dd_21"] = (running_max - cum).rolling(21).max().shift(1).abs()
        block["rvol_21"] = np.sqrt(252.0 * series.pow(2).rolling(21).mean()).shift(1)
        for j, day in enumerate(("mon", "tue", "wed", "thu", "fri")):
            block[f"dow_{day}"] = (dates.weekday == j).astype("float64")
        block["y"] = series
        block["ret_1d"] = series
        block["ticker"] = ticker
        block.index.name = "date"
        blocks.append(block.reset_index())

    out = pd.concat(blocks, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    return out.dropna()[[*TOY_FEATURE_COLUMNS, "y", "ret_1d"]].astype("float64")


def garch_returns(n: int, seed: int, *, omega: float = 3e-6) -> np.ndarray:
    """Simulate a GARCH(1, 1) return path with genuine volatility clustering.

    Args:
        n: Number of observations.
        seed: Seed for the generator.
        omega: Variance intercept.

    Returns:
        A length-``n`` array of daily log returns.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    r = np.zeros(n, dtype="float64")
    sigma2 = 1.2e-4
    for t in range(n):
        prior = r[t - 1] ** 2 if t else 1.2e-4
        sigma2 = omega + 0.10 * prior + 0.87 * sigma2
        r[t] = np.sqrt(sigma2) * shocks[t]
    return r


def split(dataset: pd.DataFrame, cut: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split a dataset into training features, training target and the window.

    Args:
        dataset: A schema-conforming dataset.
        cut: Last training date (inclusive).

    Returns:
        ``(x_train, y_train, x_window)`` where the window is every row.
    """
    dates = dataset.index.get_level_values("date")
    train = dataset.loc[dates <= cut]
    return train[TOY_FEATURE_COLUMNS], train["y"], dataset[TOY_FEATURE_COLUMNS]


def date_cut(dataset: pd.DataFrame, frac: float) -> pd.Timestamp:
    """Return the date at ``frac`` of the way through a dataset's calendar.

    Args:
        dataset: A schema-conforming dataset.
        frac: Position in ``[0, 1]``.

    Returns:
        The chosen date.
    """
    dates = dataset.index.get_level_values("date").unique().sort_values()
    return pd.Timestamp(dates[int(len(dates) * frac)])


def perturb_after(x: pd.DataFrame, cut: pd.Timestamp, shift: float = 0.5) -> pd.DataFrame:
    """Return a copy of ``x`` with every row strictly after ``cut`` mutated.

    Args:
        x: A feature frame.
        cut: Rows with a date greater than this are altered.
        shift: Amount added to every continuous feature of those rows.

    Returns:
        The mutated copy; rows on or before ``cut`` are untouched.
    """
    out = x.copy()
    future = out.index.get_level_values("date") > cut
    out.loc[future, TOY_CONTINUOUS_COLUMNS] = out.loc[future, TOY_CONTINUOUS_COLUMNS] + shift
    return out


@pytest.fixture(scope="module")
def predictable() -> pd.DataFrame:
    """A two-ticker AR(1) dataset with a signal strong enough to be found."""
    return make_synthetic_dataset(n_days=450, signal_strength=0.35, seed=11)


@pytest.fixture(scope="module")
def clustered() -> pd.DataFrame:
    """A single-ticker GARCH dataset with a deliberate high-volatility block."""
    r = garch_returns(620, seed=5)
    r[460:520] *= 5.0
    return dataset_from_returns({"VOL": r})


# --------------------------------------------------------------------------- #
# shape / dtype / index contract
# --------------------------------------------------------------------------- #
def test_arima_predict_matches_the_frame_it_was_given(predictable: pd.DataFrame) -> None:
    x_train, y_train, x = split(predictable, date_cut(predictable, 0.7))
    model = ArimaForecaster(ArimaConfig(order=(1, 0, 1)), seed=0).fit(x_train, y_train)
    pred = model.predict(x)

    assert isinstance(pred, pd.Series)
    assert pred.index.equals(x.index)
    assert pred.shape == (len(x),)
    assert pred.name == "arima"
    assert pred.dtype == np.dtype("float64")
    assert np.isfinite(pred.to_numpy()).all()
    assert model.predict_vol(x) is None
    assert model.fallback_tickers_ == []


def test_garch_predict_and_vol_match_the_frame_they_were_given(
    predictable: pd.DataFrame,
) -> None:
    x_train, y_train, x = split(predictable, date_cut(predictable, 0.7))
    model = ArimaGarchForecaster(ArimaGarchConfig(), seed=0).fit(x_train, y_train)
    pred = model.predict(x)
    vol = model.predict_vol(x)

    assert pred.index.equals(x.index)
    assert pred.name == "arima_garch"
    assert pred.dtype == np.dtype("float64")
    assert np.isfinite(pred.to_numpy()).all()

    assert vol is not None
    assert vol.index.equals(x.index)
    assert vol.shape == (len(x),)
    assert vol.name == "arima_garch"
    assert vol.dtype == np.dtype("float64")
    assert model.fallback_tickers_ == []


# --------------------------------------------------------------------------- #
# the forecasts are real one-step-ahead forecasts, not a decayed multi-step one
# --------------------------------------------------------------------------- #
def test_arima_forecasts_vary_and_track_a_predictable_series(
    predictable: pd.DataFrame,
) -> None:
    cut = date_cut(predictable, 0.7)
    x_train, y_train, x = split(predictable, cut)
    model = ArimaForecaster(ArimaConfig(order=(1, 0, 1)), seed=0).fit(x_train, y_train)
    pred = model.predict(x)

    dates = predictable.index.get_level_values("date")
    test = dates > cut
    truth = predictable.loc[test, "y"].to_numpy()
    forecast = pred[test].to_numpy()

    # Out of sample the forecast is positively correlated with what happens.
    assert np.corrcoef(forecast, truth)[0, 1] > 0.15

    # And it genuinely moves: an AR(1) with phi = 0.35 should forecast with a
    # standard deviation of roughly a third of the target's.
    assert forecast.std() > 0.2 * truth.std()

    # For contrast, the flat multi-step forecast this model deliberately avoids.
    one_ticker = y_train.xs("AAA", level="ticker")
    fitted = ARIMA(one_ticker.to_numpy(), order=(1, 0, 1), trend="c").fit()
    n_test = int((predictable.index.get_level_values("ticker") == "AAA")[test].sum())
    multi_step = np.asarray(fitted.get_forecast(n_test).predicted_mean, dtype="float64")
    assert n_test > 30
    # Ten days in it has decayed to the unconditional mean and is a flat line for
    # the remaining five months, whereas the one-step forecast keeps moving.
    assert multi_step[10:].std() < 1e-6
    assert forecast.std() > 5.0 * multi_step.std()


# --------------------------------------------------------------------------- #
# causality: nothing after t may move the prediction at t
# --------------------------------------------------------------------------- #
def test_arima_predictions_are_causal(predictable: pd.DataFrame) -> None:
    x_train, y_train, x = split(predictable, date_cut(predictable, 0.7))
    model = ArimaForecaster(ArimaConfig(order=(1, 0, 1)), seed=0).fit(x_train, y_train)

    cut = date_cut(predictable, 0.85)
    before = x.index.get_level_values("date") <= cut
    base = model.predict(x)
    perturbed = model.predict(perturb_after(x, cut))

    np.testing.assert_array_equal(base[before].to_numpy(), perturbed[before].to_numpy())
    # The perturbation has to be visible somewhere, or the test proves nothing.
    assert not np.allclose(base[~before].to_numpy(), perturbed[~before].to_numpy())


def test_garch_predictions_and_vol_are_causal(predictable: pd.DataFrame) -> None:
    x_train, y_train, x = split(predictable, date_cut(predictable, 0.7))
    model = ArimaGarchForecaster(ArimaGarchConfig(), seed=0).fit(x_train, y_train)

    cut = date_cut(predictable, 0.85)
    before = x.index.get_level_values("date") <= cut
    x_perturbed = perturb_after(x, cut)

    base, base_vol = model.predict(x), model.predict_vol(x)
    other, other_vol = model.predict(x_perturbed), model.predict_vol(x_perturbed)
    assert base_vol is not None
    assert other_vol is not None

    np.testing.assert_array_equal(base[before].to_numpy(), other[before].to_numpy())
    np.testing.assert_array_equal(base_vol[before].to_numpy(), other_vol[before].to_numpy())
    assert not np.allclose(base[~before].to_numpy(), other[~before].to_numpy())
    assert not np.allclose(base_vol[~before].to_numpy(), other_vol[~before].to_numpy())


# --------------------------------------------------------------------------- #
# order selection
# --------------------------------------------------------------------------- #
def test_arima_auto_selects_an_order_and_still_predicts(predictable: pd.DataFrame) -> None:
    x_train, y_train, x = split(predictable, date_cut(predictable, 0.7))
    config = ArimaConfig(order=(1, 0, 1), auto=True, max_p=1, max_q=1)
    model = ArimaForecaster(config, seed=0).fit(x_train, y_train)
    pred = model.predict(x)

    orders = model.selected_orders_
    assert set(orders) == {"AAA", "BBB"}
    for order in orders.values():
        assert order[1] == 0
        assert 0 <= order[0] <= 1
        assert 0 <= order[2] <= 1
    assert pred.index.equals(x.index)
    assert np.isfinite(pred.to_numpy()).all()
    assert pred.groupby(level="ticker").std().gt(0.0).all()


# --------------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------------- #
def test_garch_vol_is_positive_finite_and_of_a_daily_magnitude(
    predictable: pd.DataFrame,
) -> None:
    x_train, y_train, x = split(predictable, date_cut(predictable, 0.7))
    model = ArimaGarchForecaster(ArimaGarchConfig(), seed=0).fit(x_train, y_train)
    vol = model.predict_vol(x)
    assert vol is not None

    values = vol.to_numpy()
    assert np.isfinite(values).all()
    assert (values > 0.0).all()
    # A daily standard deviation of log returns, not an annualised one and not a
    # variance: the toy series has a 1.2% daily sigma.
    assert values.min() > 1e-3
    assert values.max() < 0.2
    assert 0.5 < float(np.median(values)) / float(predictable["y"].std()) < 2.0


def test_garch_vol_tracks_volatility_clustering(clustered: pd.DataFrame) -> None:
    calendar = pd.bdate_range(START, periods=620)
    cut = calendar[400]
    x_train, y_train, x = split(clustered, cut)
    model = ArimaGarchForecaster(ArimaGarchConfig(), seed=0).fit(x_train, y_train)
    vol = model.predict_vol(x)
    assert vol is not None

    dates = x.index.get_level_values("date")
    quiet = (dates > calendar[405]) & (dates <= calendar[455])
    loud = (dates > calendar[465]) & (dates <= calendar[520])
    assert quiet.sum() > 20
    assert loud.sum() > 20

    quiet_vol = float(vol[quiet].mean())
    loud_vol = float(vol[loud].mean())
    # The burst is a five-fold move in return size; the conditional volatility
    # has to follow it, out of sample, with parameters fixed before it happened.
    assert loud_vol > 2.0 * quiet_vol


# --------------------------------------------------------------------------- #
# graceful degradation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("cls", "config"),
    [(ArimaForecaster, ArimaConfig()), (ArimaGarchForecaster, ArimaGarchConfig())],
)
def test_a_ticker_whose_fit_fails_degrades_to_its_training_mean(
    cls: type[ArimaForecaster] | type[ArimaGarchForecaster],
    config: ArimaConfig | ArimaGarchConfig,
) -> None:
    dataset = dataset_from_returns({"FLAT": np.full(300, 0.001), "OK": garch_returns(300, seed=9)})
    x_train, y_train, x = split(dataset, date_cut(dataset, 0.8))

    model = cls(config, seed=0).fit(x_train, y_train)  # must not raise
    pred = model.predict(x)

    assert model.fallback_tickers_ == ["FLAT"]
    flat = pred.xs("FLAT", level="ticker").to_numpy()
    assert np.isfinite(flat).all()
    np.testing.assert_allclose(flat, 0.001)

    # The healthy ticker is unaffected and still gets a varying forecast.
    ok = pred.xs("OK", level="ticker").to_numpy()
    assert np.isfinite(ok).all()
    assert ok.std() > 0.0

    vol = model.predict_vol(x)
    if vol is not None:
        assert np.isfinite(vol.to_numpy()).all()
        assert (vol.to_numpy() > 0.0).all()


# --------------------------------------------------------------------------- #
# scaling, registry and misuse
# --------------------------------------------------------------------------- #
def test_forecasts_survive_a_standardised_feature_frame(predictable: pd.DataFrame) -> None:
    """The pipeline scales features; the recovered return series must be undone."""
    cut = date_cut(predictable, 0.7)
    x_train, y_train, x = split(predictable, cut)

    mean = x_train[TOY_CONTINUOUS_COLUMNS].mean()
    std = x_train[TOY_CONTINUOUS_COLUMNS].std(ddof=0)
    scaled_train = x_train.copy()
    scaled = x.copy()
    scaled_train[TOY_CONTINUOUS_COLUMNS] = (x_train[TOY_CONTINUOUS_COLUMNS] - mean) / std
    scaled[TOY_CONTINUOUS_COLUMNS] = (x[TOY_CONTINUOUS_COLUMNS] - mean) / std

    raw_model = ArimaForecaster(ArimaConfig(order=(1, 0, 1)), seed=0).fit(x_train, y_train)
    scaled_model = ArimaForecaster(ArimaConfig(order=(1, 0, 1)), seed=0).fit(scaled_train, y_train)
    np.testing.assert_allclose(
        raw_model.predict(x).to_numpy(), scaled_model.predict(scaled).to_numpy(), atol=1e-10
    )

    scaled_garch = ArimaGarchForecaster(ArimaGarchConfig(), seed=0).fit(scaled_train, y_train)
    scaled_vol = scaled_garch.predict_vol(scaled)
    assert scaled_vol is not None
    assert scaled_vol.max() < 0.2


def test_build_model_finds_both_forecasters() -> None:
    arima = build_model("arima", ArimaConfig(), seed=3)
    garch = build_model("arima_garch", ArimaGarchConfig(), seed=3)

    assert isinstance(arima, ArimaForecaster)
    assert isinstance(garch, ArimaGarchForecaster)
    assert arima.name == "arima"
    assert garch.name == "arima_garch"


def test_predicting_before_fitting_raises(predictable: pd.DataFrame) -> None:
    x = predictable[TOY_FEATURE_COLUMNS]
    with pytest.raises(NotFittedError):
        ArimaForecaster(ArimaConfig(), seed=0).predict(x)
    with pytest.raises(NotFittedError):
        ArimaGarchForecaster(ArimaGarchConfig(), seed=0).predict(x)
    with pytest.raises(NotFittedError):
        ArimaGarchForecaster(ArimaGarchConfig(), seed=0).predict_vol(x)


@pytest.mark.parametrize(
    "config",
    [
        ArimaConfig(order=(1, 1, 1)),  # d must be 0: the model is fitted on returns
        ArimaConfig(order=(-1, 0, 1)),
        ArimaConfig(trend="quadratic"),
        ArimaConfig(max_p=-1),
    ],
)
def test_invalid_arima_config_is_rejected_at_construction(config: ArimaConfig) -> None:
    with pytest.raises(ValueError, match="arima"):
        ArimaForecaster(config, seed=0)


@pytest.mark.parametrize(
    "config",
    [
        ArimaGarchConfig(dist="cauchy"),
        ArimaGarchConfig(p=0),
        ArimaGarchConfig(mean_lags=-1),
    ],
)
def test_invalid_garch_config_is_rejected_at_construction(config: ArimaGarchConfig) -> None:
    with pytest.raises(ValueError, match="arima_garch"):
        ArimaGarchForecaster(config, seed=0)
