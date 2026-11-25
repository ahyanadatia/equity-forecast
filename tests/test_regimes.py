"""Tests for volatility-regime labelling and per-regime scoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from equity_forecast.regimes import (
    REGIME_METRIC_COLUMNS,
    holm_adjust,
    label_regimes,
    market_vol_signal,
    regime_metrics,
)
from equity_forecast.splits import Fold

DATES = pd.bdate_range("2020-01-01", periods=120, name="date")


def _fold(train: slice, test: slice, fold_id: int = 0) -> Fold:
    return Fold(
        fold_id=fold_id,
        train_start=DATES[train.start],
        train_end=DATES[train.stop - 1],
        test_start=DATES[test.start],
        test_end=DATES[test.stop - 1],
    )


def test_market_vol_signal_is_the_cross_sectional_mean() -> None:
    index = pd.MultiIndex.from_product([DATES[:3], ["A", "B"]], names=["date", "ticker"])
    frame = pd.DataFrame({"rvol_21": [0.1, 0.3, 0.2, 0.4, 0.5, 0.7]}, index=index)
    signal = market_vol_signal(frame, "rvol_21")
    np.testing.assert_allclose(signal.to_numpy(), [0.2, 0.3, 0.6])
    with pytest.raises(KeyError):
        market_vol_signal(frame, "rvol_63")


def test_cut_points_come_from_training_dates_only() -> None:
    # Training signal is 0..59; the test block is far above it.
    signal = pd.Series(np.arange(120, dtype=float), index=DATES)
    signal.iloc[70:] = 1000.0
    labels = label_regimes(signal, [_fold(slice(0, 60), slice(70, 120))])
    expected_low, expected_high = np.quantile(np.arange(60.0), [1 / 3, 2 / 3])
    assert labels["low_cut"].iloc[0] == pytest.approx(expected_low)
    assert labels["high_cut"].iloc[0] == pytest.approx(expected_high)
    # Relative to its training history, the whole test block is high volatility.
    assert set(labels["regime"]) == {"high"}
    assert len(labels) == 50


def test_mutating_the_test_block_does_not_move_the_cut_points() -> None:
    rng = np.random.default_rng(0)
    signal = pd.Series(rng.uniform(0.1, 0.4, 120), index=DATES)
    fold = _fold(slice(0, 60), slice(65, 120))
    base = label_regimes(signal, [fold])
    shocked = signal.copy()
    shocked.iloc[65:] *= 3.0
    moved = label_regimes(shocked, [fold])
    assert base["low_cut"].iloc[0] == moved["low_cut"].iloc[0]
    assert base["high_cut"].iloc[0] == moved["high_cut"].iloc[0]


def test_bad_quantiles_are_rejected() -> None:
    signal = pd.Series(np.ones(120), index=DATES)
    with pytest.raises(ValueError, match="quantiles"):
        label_regimes(signal, [_fold(slice(0, 60), slice(65, 120))], quantiles=(0.7, 0.3))


def test_holm_matches_a_hand_computed_example() -> None:
    adjusted = holm_adjust([0.01, 0.04, math.nan, 0.03])
    # Sorted 0.01, 0.03, 0.04 with m=3 -> 0.03, 0.06, max(0.06, 0.04) = 0.06.
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[3] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert math.isnan(adjusted[2])


def _toy_run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(1)
    test_dates = DATES[60:]
    rows = []
    for date in test_dates:
        for ticker in ("A", "B"):
            y = float(rng.normal(0, 0.01))
            rows.append((date, ticker, 0, "zero_return", 0.0, y))
            rows.append((date, ticker, 0, "oracle", y, y))
    forecasts = pd.DataFrame(
        rows, columns=["date", "ticker", "fold_id", "model", "forecast", "y_true"]
    )
    labels = pd.DataFrame(
        {
            "date": test_dates,
            "fold_id": 0,
            "signal": 0.0,
            "low_cut": 0.0,
            "high_cut": 0.0,
            "regime": np.where(np.arange(len(test_dates)) % 2 == 0, "high", "low"),
        }
    )
    pnl = pd.DataFrame(
        {
            "date": np.tile(test_dates, 2),
            "model": ["zero_return"] * len(test_dates) + ["oracle"] * len(test_dates),
            "net": np.r_[np.zeros(len(test_dates)), rng.normal(0.001, 0.01, len(test_dates))],
        }
    )
    return forecasts, labels, pnl


def test_regime_metrics_scores_each_model_in_each_regime() -> None:
    forecasts, labels, pnl = _toy_run()
    benchmark = pd.Series(0.0005, index=labels["date"])
    table = regime_metrics(
        forecasts,
        labels,
        pnl,
        ["zero_return", "oracle"],
        baseline="zero_return",
        horizon=1,
        ann_factor=252,
        benchmark_net=benchmark,
    )
    assert list(table.columns) == list(REGIME_METRIC_COLUMNS)
    oracle_high = table.loc[(table["regime"] == "high") & (table["model"] == "oracle")].iloc[0]
    assert oracle_high["n_dates"] == 30
    assert oracle_high["rmse"] == pytest.approx(0.0)
    assert oracle_high["dm_stat_vs_zero"] < 0
    assert oracle_high["dm_p_holm"] < 0.05
    base = table.loc[table["model"] == "zero_return"]
    assert base["dm_p_vs_zero"].isna().all()
    assert (table["model"] == "buy_and_hold").sum() == 3
    # "mid" has no dates in this fixture, so its rows are empty rather than invented.
    assert (table.loc[table["regime"] == "mid", "n_dates"] == 0).all()


def test_empty_inputs_give_an_empty_table() -> None:
    empty = regime_metrics(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        ["zero_return"],
        baseline="zero_return",
        horizon=1,
        ann_factor=252,
    )
    assert empty.empty
