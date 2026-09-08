"""Tests for the two return baselines and the ridge regression.

These cover ``models/naive.py`` and ``models/linear.py`` specifically; the
cross-model contract suite lives in ``tests/test_models.py``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from equity_forecast.config import RidgeConfig
from equity_forecast.models.base import BaseForecaster, build_model
from equity_forecast.models.linear import Ridge
from equity_forecast.models.naive import HistoricalMean, NotFittedError, ZeroReturn

MODEL_CASES: list[tuple[str, Any, type[BaseForecaster]]] = [
    ("zero_return", {}, ZeroReturn),
    ("historical_mean", {}, HistoricalMean),
    ("ridge", RidgeConfig(), Ridge),
]
CASE_IDS = [case[0] for case in MODEL_CASES]


class Fold(NamedTuple):
    """One train/test split of a toy dataset, with train-fold scaling applied."""

    x: pd.DataFrame
    y: pd.Series
    train: np.ndarray
    test: np.ndarray


def make_fold(dataset: pd.DataFrame, train_frac: float = 0.7) -> Fold:
    """Split by date and standardise the continuous columns on train statistics."""
    x_raw = dataset.drop(columns=["y", "ret_1d"])
    y = dataset["y"]
    dates = dataset.index.get_level_values("date")
    cutoff = dates.unique().sort_values()[int(train_frac * dates.nunique())]
    train = np.asarray(dates < cutoff)
    test = np.asarray(dates >= cutoff)

    continuous = [c for c in x_raw.columns if not str(c).startswith("dow_")]
    mean = x_raw.loc[train, continuous].mean()
    std = x_raw.loc[train, continuous].std(ddof=0).replace(0.0, 1.0)
    x = x_raw.copy()
    x[continuous] = (x_raw[continuous] - mean) / std
    return Fold(x=x, y=y, train=train, test=test)


def fit_case(params: Any, cls: type[BaseForecaster], fold: Fold) -> BaseForecaster:
    """Build a model from the registry and fit it on the fold's training rows."""
    model = build_model(cls.name, params, seed=13)
    model.fit(fold.x[fold.train], fold.y[fold.train])
    return model


# --------------------------------------------------------------------------- #
# shape / index / naming contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "params", "cls"), MODEL_CASES, ids=CASE_IDS)
def test_predictions_preserve_index_dtype_and_name(
    synthetic_dataset: pd.DataFrame, name: str, params: Any, cls: type[BaseForecaster]
) -> None:
    fold = make_fold(synthetic_dataset)
    model = fit_case(params, cls, fold)
    pred = model.predict(fold.x)

    assert isinstance(pred, pd.Series)
    assert pred.index.equals(fold.x.index)
    assert list(pred.index) == list(fold.x.index)
    assert pred.index.names == fold.x.index.names
    assert pred.dtype == np.float64
    assert pred.name == name
    assert len(pred) == len(fold.x)
    assert pred.notna().all()


@pytest.mark.parametrize(("name", "params", "cls"), MODEL_CASES, ids=CASE_IDS)
def test_predict_vol_is_none(
    synthetic_dataset: pd.DataFrame, name: str, params: Any, cls: type[BaseForecaster]
) -> None:
    del name
    fold = make_fold(synthetic_dataset)
    model = fit_case(params, cls, fold)
    assert model.predict_vol(fold.x) is None


@pytest.mark.parametrize(("name", "params", "cls"), MODEL_CASES, ids=CASE_IDS)
def test_build_model_resolves_every_registry_key(
    name: str, params: Any, cls: type[BaseForecaster]
) -> None:
    model = build_model(name, params, seed=0)
    assert isinstance(model, cls)
    assert model.name == name


@pytest.mark.parametrize(("name", "params", "cls"), MODEL_CASES, ids=CASE_IDS)
def test_predict_before_fit_raises(
    synthetic_dataset: pd.DataFrame, name: str, params: Any, cls: type[BaseForecaster]
) -> None:
    del cls
    model = build_model(name, params, seed=0)
    x = synthetic_dataset.drop(columns=["y", "ret_1d"])
    with pytest.raises(NotFittedError, match="before fit"):
        model.predict(x)
    # The error is catchable as either of the two natural built-in kinds.
    with pytest.raises((RuntimeError, ValueError)):
        model.predict(x)


@pytest.mark.parametrize(("name", "params", "cls"), MODEL_CASES, ids=CASE_IDS)
def test_predictions_are_causal(
    synthetic_dataset: pd.DataFrame, name: str, params: Any, cls: type[BaseForecaster]
) -> None:
    del name
    fold = make_fold(synthetic_dataset)
    model = fit_case(params, cls, fold)
    baseline = model.predict(fold.x)

    dates = fold.x.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    split = unique_dates[int(0.9 * len(unique_dates))]
    future = np.asarray(dates >= split)

    tampered = fold.x.copy()
    tampered.loc[future, :] = 999.0
    after = model.predict(tampered)

    pd.testing.assert_series_equal(baseline[~future], after[~future])


# --------------------------------------------------------------------------- #
# ZeroReturn
# --------------------------------------------------------------------------- #


def test_zero_return_predicts_exactly_zero(synthetic_dataset: pd.DataFrame) -> None:
    fold = make_fold(synthetic_dataset)
    pred = ZeroReturn({}, 0).fit(fold.x[fold.train], fold.y[fold.train]).predict(fold.x)

    assert (pred == 0.0).all()
    assert np.array_equal(pred.to_numpy(), np.zeros(len(fold.x)))
    assert pred.abs().max() == 0.0


def test_zero_return_ignores_the_training_data(predictable_dataset: pd.DataFrame) -> None:
    # A genuinely forecastable dataset must not move the baseline at all.
    fold = make_fold(predictable_dataset)
    model = ZeroReturn({}, 0).fit(fold.x[fold.train], fold.y[fold.train])
    assert (model.predict(fold.x) == 0.0).all()


# --------------------------------------------------------------------------- #
# HistoricalMean
# --------------------------------------------------------------------------- #


def test_historical_mean_equals_per_ticker_train_mean(synthetic_dataset: pd.DataFrame) -> None:
    fold = make_fold(synthetic_dataset)
    model = HistoricalMean({}, 0).fit(fold.x[fold.train], fold.y[fold.train])
    pred = model.predict(fold.x)

    expected = fold.y[fold.train].groupby(level="ticker").mean()
    assert len(expected) > 1
    assert expected.nunique() == len(expected)  # the tickers really do differ

    # Constant per ticker, and equal to that ticker's training mean.
    spread = pred.groupby(level="ticker").agg(lambda s: float(s.max() - s.min()))
    assert (spread == 0.0).all()
    for ticker, mean in expected.items():
        got = pred.xs(ticker, level="ticker")
        assert np.allclose(got.to_numpy(), mean)

    pd.testing.assert_series_equal(
        model.ticker_means_.sort_index(),
        expected.sort_index().rename("mean_return").astype("float64"),
    )


def test_historical_mean_unseen_ticker_falls_back_to_pooled_mean(
    synthetic_dataset: pd.DataFrame,
) -> None:
    fold = make_fold(synthetic_dataset)
    model = HistoricalMean({}, 0).fit(fold.x[fold.train], fold.y[fold.train])

    per_ticker = fold.y[fold.train].groupby(level="ticker").mean()
    pooled = float(fold.y[fold.train].mean())
    assert model.pooled_mean_ == pytest.approx(pooled)

    # Rename one ticker to a name the model has never seen.
    unseen = fold.x.xs("BBB", level="ticker", drop_level=False).rename(
        index={"BBB": "ZZZ"}, level="ticker"
    )
    pred = model.predict(unseen)

    assert pred.index.equals(unseen.index)
    assert np.allclose(pred.to_numpy(), pooled)
    # The fallback is genuinely the pooled mean, not some ticker's own mean.
    for mean in per_ticker:
        assert not np.isclose(pooled, mean)


# --------------------------------------------------------------------------- #
# Ridge
# --------------------------------------------------------------------------- #


def test_ridge_recovers_an_ar1_signal(predictable_dataset: pd.DataFrame) -> None:
    fold = make_fold(predictable_dataset)
    model = Ridge(RidgeConfig(), 13).fit(fold.x[fold.train], fold.y[fold.train])
    pred = model.predict(fold.x)

    truth = fold.y[fold.test]
    forecast = pred[fold.test]
    assert len(truth) > 100

    non_zero = truth != 0.0
    directional_accuracy = float((np.sign(forecast) == np.sign(truth))[non_zero].mean())
    correlation = float(np.corrcoef(forecast.to_numpy(), truth.to_numpy())[0, 1])

    assert directional_accuracy > 0.55
    assert correlation > 0.15

    # The same model finds nothing to hold on to in the unpredictable dataset.
    noise_fold = make_fold(
        predictable_dataset.assign(y=lambda f: f["y"].sample(frac=1.0, random_state=3).to_numpy())
    )
    noise_model = Ridge(RidgeConfig(), 13).fit(
        noise_fold.x[noise_fold.train], noise_fold.y[noise_fold.train]
    )
    noise_pred = noise_model.predict(noise_fold.x)[noise_fold.test]
    noise_truth = noise_fold.y[noise_fold.test]
    noise_corr = float(np.corrcoef(noise_pred.to_numpy(), noise_truth.to_numpy())[0, 1])
    assert abs(noise_corr) < correlation


def test_ridge_huge_alpha_shrinks_predictions_towards_zero(
    predictable_dataset: pd.DataFrame,
) -> None:
    fold = make_fold(predictable_dataset)
    light = Ridge(RidgeConfig(alpha=1.0), 13).fit(fold.x[fold.train], fold.y[fold.train])
    heavy = Ridge(RidgeConfig(alpha=1e9), 13).fit(fold.x[fold.train], fold.y[fold.train])

    light_pred = light.predict(fold.x)
    heavy_pred = heavy.predict(fold.x)

    assert float(heavy_pred.std()) < 1e-3 * float(light_pred.std())
    assert float(heavy.coefficients_.abs().max()) < 1e-3 * float(light.coefficients_.abs().max())
    # What survives is the intercept, i.e. the historical-mean baseline.
    assert float((heavy_pred - heavy.intercept_).abs().max()) < 1e-6
    assert heavy.intercept_ == pytest.approx(float(fold.y[fold.train].mean()), abs=1e-9)

    # Without an intercept the shrunken model collapses onto zero itself.
    flat = Ridge(RidgeConfig(alpha=1e9, fit_intercept=False), 13).fit(
        fold.x[fold.train], fold.y[fold.train]
    )
    assert float(flat.predict(fold.x).abs().max()) < 1e-6


def test_ridge_coefficients_are_inspectable(predictable_dataset: pd.DataFrame) -> None:
    fold = make_fold(predictable_dataset)
    model = Ridge(RidgeConfig(), 13)

    with pytest.raises(NotFittedError):
        _ = model.coefficients_

    model.fit(fold.x[fold.train], fold.y[fold.train])
    coefficients = model.coefficients_

    assert isinstance(coefficients, pd.Series)
    assert list(coefficients.index) == list(fold.x.columns)
    assert coefficients.dtype == np.float64
    assert np.isfinite(coefficients.to_numpy()).all()
    # The AR(1) driver is the dominant feature, with the right (positive) sign.
    assert coefficients.abs().idxmax() == "ret_lag_1"
    assert coefficients["ret_lag_1"] > 0.0


def test_ridge_handles_non_finite_rows(synthetic_dataset: pd.DataFrame) -> None:
    fold = make_fold(synthetic_dataset)
    x = fold.x.copy()
    x.iloc[0, 0] = np.nan
    x.iloc[5, 2] = np.inf
    x.iloc[-1, 1] = np.nan

    clean = Ridge(RidgeConfig(), 13).fit(fold.x[fold.train], fold.y[fold.train])
    dirty = Ridge(RidgeConfig(), 13).fit(x[fold.train], fold.y[fold.train])

    pred = dirty.predict(x)
    assert pred.index.equals(x.index)
    assert np.isfinite(pred.to_numpy()).all()
    assert np.isfinite(dirty.coefficients_.to_numpy()).all()
    # Two poisoned training rows out of hundreds barely move the fit.
    assert np.allclose(dirty.coefficients_.to_numpy(), clean.coefficients_.to_numpy(), atol=1e-3)


def test_ridge_rejects_a_missing_feature_column(synthetic_dataset: pd.DataFrame) -> None:
    fold = make_fold(synthetic_dataset)
    model = Ridge(RidgeConfig(), 13).fit(fold.x[fold.train], fold.y[fold.train])
    with pytest.raises(ValueError, match="missing feature column"):
        model.predict(fold.x.drop(columns=["ret_lag_1"]))
