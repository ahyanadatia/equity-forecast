"""Tests for the feature/dataset builder.

The load-bearing tests here are the causality ones: ``test_no_look_ahead_*``
mutates the future of the price panel and demands that already-built rows come
back bitwise identical (NFR-2), and ``test_cross_ticker_isolation`` demands the
same across tickers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from equity_forecast.config import FeatureConfig
from equity_forecast.features import (
    DAY_OF_WEEK_COLUMNS,
    FEATURE_SPLIT,
    apply_scaler,
    binary_feature_columns,
    build_dataset,
    continuous_feature_columns,
    feature_columns,
    fit_scaler,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

# 14 daily log returns; index i of this list is the return of day i + 1.
TINY_RETURNS: list[float] = [
    0.01,
    -0.02,
    0.03,
    -0.01,
    0.02,
    -0.03,
    0.01,
    0.005,
    -0.015,
    0.02,
    0.0,
    0.01,
    -0.005,
    0.03,
]

TINY_CFG = FeatureConfig(
    horizon=1,
    n_lags=2,
    roll_windows=(3,),
    drawdown_window=3,
    rvol_window=3,
    include_day_of_week=True,
)

PANEL_CFG = FeatureConfig(
    horizon=1,
    n_lags=3,
    roll_windows=(5, 21),
    drawdown_window=21,
    rvol_window=21,
    include_day_of_week=True,
)


def panel_from_returns(
    returns: Mapping[str, Sequence[float] | npt.NDArray[np.float64]],
    start: str = "2020-01-01",
    first_price: float = 100.0,
) -> pd.DataFrame:
    """Build a price panel whose log returns are exactly ``returns``."""
    n_days = len(next(iter(returns.values()))) + 1
    dates = pd.bdate_range(start, periods=n_days, name="date")
    data = {
        ticker: first_price * np.exp(np.concatenate([[0.0], np.cumsum(np.asarray(r, dtype=float))]))
        for ticker, r in returns.items()
    }
    panel = pd.DataFrame(data, index=dates)
    panel.columns.name = "ticker"
    return panel


def tiny_panel(ticker: str = "AAA") -> pd.DataFrame:
    """A 15-day, single-ticker panel with hand-chosen log returns."""
    return panel_from_returns({ticker: TINY_RETURNS})


def dates_of(dataset: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the ``date`` level of a dataset index."""
    return pd.DatetimeIndex(dataset.index.get_level_values("date"))


def regime_panel() -> pd.DataFrame:
    """A two-ticker panel whose volatility jumps 6x after 200 days."""
    rng = np.random.default_rng(20240921)
    calm = rng.normal(0.0, 0.01, size=(200, 2))
    storm = rng.normal(0.0, 0.06, size=(100, 2))
    returns = np.vstack([calm, storm])
    return panel_from_returns({"AAA": returns[:, 0], "BBB": returns[:, 1]})


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_dataset_schema_is_exact(price_panel: pd.DataFrame) -> None:
    dataset = build_dataset(price_panel, PANEL_CFG)

    assert list(dataset.columns) == [
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "roll_mean_5",
        "roll_std_5",
        "roll_mean_21",
        "roll_std_21",
        "max_dd_21",
        "rvol_21",
        "dow_mon",
        "dow_tue",
        "dow_wed",
        "dow_thu",
        "dow_fri",
        "y",
        "ret_1d",
    ]
    assert list(dataset.columns) == [*feature_columns(PANEL_CFG), "y", "ret_1d"]
    assert isinstance(dataset.index, pd.MultiIndex)
    assert dataset.index.names == ["date", "ticker"]
    assert dataset.index.is_monotonic_increasing
    assert not dataset.index.has_duplicates
    assert set(dataset.index.get_level_values("ticker")) == {"SYNA", "SYNB"}
    assert not bool(dataset.isna().to_numpy().any())
    assert all(dtype == np.dtype("float64") for dtype in dataset.dtypes)
    assert bool(np.isfinite(dataset.to_numpy()).all())
    # horizon 1: the target is just the realised return of the row's own date.
    pd.testing.assert_series_equal(dataset["y"], dataset["ret_1d"], check_names=False)


def test_feature_columns_partition_and_respect_config() -> None:
    continuous = continuous_feature_columns(PANEL_CFG)
    binary = binary_feature_columns(PANEL_CFG)

    assert continuous + binary == feature_columns(PANEL_CFG)
    assert binary == list(DAY_OF_WEEK_COLUMNS)
    assert not any(c.startswith(FEATURE_SPLIT["binary"]) for c in continuous)
    assert all(c.startswith(FEATURE_SPLIT["continuous"]) for c in continuous)

    no_dow = FeatureConfig(n_lags=2, roll_windows=(4,), include_day_of_week=False)
    assert binary_feature_columns(no_dow) == []
    assert feature_columns(no_dow) == [
        "ret_lag_1",
        "ret_lag_2",
        "roll_mean_4",
        "roll_std_4",
        "max_dd_21",
        "rvol_21",
    ]
    assert continuous_feature_columns(no_dow) == feature_columns(no_dow)


# --------------------------------------------------------------------------- #
# known answers, computed by hand
# --------------------------------------------------------------------------- #


def test_known_answer_on_hand_built_panel() -> None:
    dataset = build_dataset(tiny_panel(), TINY_CFG)

    # 15 prices -> returns r_1..r_14. The first row needs r_{t-3}, so t = 4 is the
    # first complete row and every later row survives (horizon 1).
    assert len(dataset) == 11
    assert dates_of(dataset)[0] == pd.Timestamp("2020-01-07")

    # Row t = 2020-01-13 is the 9th date (t = 8, zero-based), a Monday.
    # Its window of three returns ending at t-1 is r_5, r_6, r_7 = 0.02, -0.03, 0.01.
    row = dataset.loc[(pd.Timestamp("2020-01-13"), "AAA")]

    assert row["ret_lag_1"] == pytest.approx(0.01, abs=1e-12)  # r_7
    assert row["ret_lag_2"] == pytest.approx(-0.03, abs=1e-12)  # r_6
    assert row["roll_mean_3"] == pytest.approx((0.02 - 0.03 + 0.01) / 3.0, abs=1e-12)
    # ddof=1 about a window mean of exactly zero:
    assert row["roll_std_3"] == pytest.approx(
        math.sqrt((0.02**2 + 0.03**2 + 0.01**2) / 2.0), abs=1e-12
    )
    assert row["rvol_3"] == pytest.approx(
        math.sqrt(252.0 * (0.02**2 + 0.03**2 + 0.01**2) / 3.0), abs=1e-12
    )
    # cumulative path 0.02, -0.01, 0.00; running peak (floored at 0) 0.02, 0.02, 0.02;
    # worst log drawdown -0.03 at the second step.
    assert row["max_dd_3"] == pytest.approx(1.0 - math.exp(-0.03), abs=1e-12)
    assert row["ret_1d"] == pytest.approx(0.005, abs=1e-12)  # r_8
    assert row["y"] == pytest.approx(0.005, abs=1e-12)
    assert row["dow_mon"] == 1.0
    assert row[list(DAY_OF_WEEK_COLUMNS)].sum() == 1.0


def test_known_answer_drawdown_with_interior_peak() -> None:
    dataset = build_dataset(tiny_panel(), TINY_CFG)

    # Row t = 2020-01-16 (t = 11). Window r_8, r_9, r_10 = 0.005, -0.015, 0.02.
    # cumulative 0.005, -0.010, 0.010; running peak 0.005, 0.005, 0.010;
    # worst log drawdown -0.015 -> the peak that matters is inside the window, not the start.
    row = dataset.loc[(pd.Timestamp("2020-01-16"), "AAA")]
    assert row["max_dd_3"] == pytest.approx(1.0 - math.exp(-0.015), abs=1e-12)
    assert row["roll_mean_3"] == pytest.approx(0.01 / 3.0, abs=1e-12)
    assert 0.0 <= row["max_dd_3"] < 1.0


def test_max_drawdown_is_zero_for_a_monotonically_rising_path() -> None:
    panel = panel_from_returns({"AAA": [0.004] * 12})
    dataset = build_dataset(panel, TINY_CFG)
    assert dataset["max_dd_3"].to_numpy() == pytest.approx(0.0, abs=1e-12)


def test_ret_lag_1_is_the_previous_rows_realised_return(price_panel: pd.DataFrame) -> None:
    dataset = build_dataset(price_panel, PANEL_CFG)
    for ticker in ("SYNA", "SYNB"):
        block = dataset.xs(ticker, level="ticker")
        np.testing.assert_array_equal(
            block["ret_lag_1"].to_numpy()[1:], block["ret_1d"].to_numpy()[:-1]
        )


# --------------------------------------------------------------------------- #
# horizon
# --------------------------------------------------------------------------- #


def test_horizon_three_sums_three_returns_and_drops_the_tail() -> None:
    panel = tiny_panel()
    cfg_h1 = TINY_CFG
    cfg_h3 = FeatureConfig(
        horizon=3,
        n_lags=2,
        roll_windows=(3,),
        drawdown_window=3,
        rvol_window=3,
        include_day_of_week=True,
    )
    base = build_dataset(panel, cfg_h1)
    horizon = build_dataset(panel, cfg_h3)

    # The last h-1 = 2 rows have no complete target and must be gone.
    assert len(horizon) == len(base) - 2
    assert dates_of(horizon).max() == dates_of(base)[-3]

    returns = base["ret_1d"].to_numpy()
    expected = returns[:-2] + returns[1:-1] + returns[2:]
    np.testing.assert_allclose(horizon["y"].to_numpy(), expected, rtol=0, atol=1e-12)

    # Only the target changes with the horizon; every feature is identical.
    features = feature_columns(cfg_h3)
    pd.testing.assert_frame_equal(
        horizon[features], base.loc[horizon.index, features], check_exact=True
    )


# --------------------------------------------------------------------------- #
# causality (NFR-2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cut_position", [40, 120, 300, 600])
def test_no_look_ahead_when_the_future_is_mutated(
    price_panel: pd.DataFrame, cut_position: int
) -> None:
    cut = price_panel.index[cut_position]
    base = build_dataset(price_panel, PANEL_CFG)

    rng = np.random.default_rng(cut_position)
    mutated = price_panel.copy()
    future = mutated.index > cut
    shock = rng.uniform(0.4, 2.5, size=(int(future.sum()), mutated.shape[1]))
    mutated.loc[future] = mutated.loc[future].to_numpy() * shock
    assert not mutated.equals(price_panel)

    rebuilt = build_dataset(mutated, PANEL_CFG)

    before = base[dates_of(base) <= cut]
    after = rebuilt[dates_of(rebuilt) <= cut]
    assert len(before) > 0
    # Bitwise: not a single feature (nor the already-realised target) may move.
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_cross_ticker_isolation() -> None:
    rng = np.random.default_rng(5)
    returns_a = rng.normal(0.0, 0.01, 120)
    returns_b = rng.normal(0.0, 0.01, 120)
    base = build_dataset(panel_from_returns({"AAA": returns_a, "BBB": returns_b}), PANEL_CFG)

    other_b = rng.normal(0.02, 0.09, 120)
    changed = build_dataset(panel_from_returns({"AAA": returns_a, "BBB": other_b}), PANEL_CFG)

    block_before = base.xs("AAA", level="ticker")
    block_after = changed.xs("AAA", level="ticker")
    assert len(block_before) > 0
    pd.testing.assert_frame_equal(block_before, block_after, check_exact=True)
    assert not base.xs("BBB", level="ticker").equals(changed.xs("BBB", level="ticker"))


def test_day_of_week_one_hot_sums_to_exactly_one(price_panel: pd.DataFrame) -> None:
    dataset = build_dataset(price_panel, PANEL_CFG)
    dow = dataset[list(DAY_OF_WEEK_COLUMNS)]

    np.testing.assert_array_equal(dow.sum(axis=1).to_numpy(), np.ones(len(dataset)))
    assert set(np.unique(dow.to_numpy())) == {0.0, 1.0}

    weekday = np.asarray(dates_of(dataset).weekday)
    for position, name in enumerate(DAY_OF_WEEK_COLUMNS):
        np.testing.assert_array_equal(dow[name].to_numpy(), (weekday == position).astype(float))


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #


def test_fit_scaler_uses_train_rows_only() -> None:
    dataset = build_dataset(regime_panel(), PANEL_CFG)
    split = dates_of(dataset).unique()[150]
    train = dataset[dates_of(dataset) <= split]
    test = dataset[dates_of(dataset) > split]
    assert len(train) > 0 and len(test) > 0

    scaler = fit_scaler(train, PANEL_CFG)
    scaled = apply_scaler(scaler, dataset)
    continuous = continuous_feature_columns(PANEL_CFG)

    train_block = scaled[dates_of(scaled) <= split][continuous].to_numpy()
    np.testing.assert_allclose(train_block.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(train_block.std(axis=0, ddof=0), 1.0, atol=1e-10)

    # The test period is six times as volatile. If its statistics had leaked into the
    # fit, these columns would also come out at unit scale; they must not.
    test_block = scaled[dates_of(scaled) > split][continuous]
    assert test_block["ret_lag_1"].std(ddof=0) > 2.0
    assert test_block["rvol_21"].std(ddof=0) > 2.0

    # And the scaler really is the train-only one, applied verbatim.
    raw_train = train[continuous].to_numpy()
    np.testing.assert_array_equal(
        scaled.loc[train.index, continuous].to_numpy(),
        (raw_train - np.asarray(scaler.mean)) / np.asarray(scaler.std),
    )
    full_scaler = fit_scaler(dataset, PANEL_CFG)
    assert full_scaler.std != scaler.std


def test_apply_scaler_leaves_binary_and_target_columns_untouched() -> None:
    dataset = build_dataset(regime_panel(), PANEL_CFG)
    scaler = fit_scaler(dataset, PANEL_CFG)
    scaled = apply_scaler(scaler, dataset)

    assert list(scaled.columns) == list(dataset.columns)
    assert scaled.index.equals(dataset.index)
    assert all(dtype == np.dtype("float64") for dtype in scaled.dtypes)
    untouched = [*binary_feature_columns(PANEL_CFG), "y", "ret_1d"]
    pd.testing.assert_frame_equal(scaled[untouched], dataset[untouched], check_exact=True)
    assert not scaled[continuous_feature_columns(PANEL_CFG)].equals(
        dataset[continuous_feature_columns(PANEL_CFG)]
    )
    # The input frame is not modified in place.
    assert dataset["ret_lag_1"].std(ddof=0) < 0.5

    assert list(scaler.to_frame().index) == continuous_feature_columns(PANEL_CFG)


def test_zero_variance_columns_scale_without_nan_or_inf() -> None:
    # Constant log returns -> every continuous feature is constant across rows.
    panel = panel_from_returns({"AAA": [0.001] * 60})
    dataset = build_dataset(panel, PANEL_CFG)
    continuous = continuous_feature_columns(PANEL_CFG)
    assert dataset[continuous].std(ddof=0).max() < 1e-12

    scaler = fit_scaler(dataset, PANEL_CFG)
    assert scaler.std == tuple([1.0] * len(continuous))

    scaled = apply_scaler(scaler, dataset)
    values = scaled[continuous].to_numpy()
    assert bool(np.isfinite(values).all())
    assert not bool(np.isnan(values).any())
    np.testing.assert_allclose(values, 0.0, atol=1e-9)


def test_scaler_input_validation() -> None:
    dataset = build_dataset(regime_panel(), PANEL_CFG)
    scaler = fit_scaler(dataset, PANEL_CFG)

    with pytest.raises(KeyError, match="ret_lag_1"):
        fit_scaler(dataset.drop(columns=["ret_lag_1"]), PANEL_CFG)
    with pytest.raises(ValueError, match="empty"):
        fit_scaler(dataset.iloc[:0], PANEL_CFG)
    with pytest.raises(ValueError, match="NaN"):
        fit_scaler(dataset.assign(rvol_21=np.nan), PANEL_CFG)
    with pytest.raises(KeyError, match="rvol_21"):
        apply_scaler(scaler, dataset.drop(columns=["rvol_21"]))


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"n_lags": 0},
        {"roll_windows": (1, 5)},
        {"roll_windows": (5, 5)},
        {"drawdown_window": 0},
        {"rvol_window": 0},
    ],
)
def test_invalid_feature_config_is_rejected(kwargs: dict[str, Any]) -> None:
    cfg = FeatureConfig(**kwargs)
    with pytest.raises(ValueError, match=r"features\."):
        feature_columns(cfg)
    with pytest.raises(ValueError, match=r"features\."):
        build_dataset(tiny_panel(), cfg)


def test_invalid_price_panel_is_rejected() -> None:
    panel = tiny_panel()

    with pytest.raises(TypeError, match="DatetimeIndex"):
        build_dataset(panel.reset_index(drop=True), TINY_CFG)
    with pytest.raises(ValueError, match="non-positive"):
        build_dataset(panel.assign(AAA=-1.0), TINY_CFG)
    with pytest.raises(ValueError, match="no ticker columns"):
        build_dataset(panel.drop(columns=["AAA"]), TINY_CFG)
    with pytest.raises(ValueError, match="duplicate dates"):
        build_dataset(pd.concat([panel, panel.iloc[-1:]]), TINY_CFG)


# --------------------------------------------------------------------------- #
# optional cross-sectional (market) features
# --------------------------------------------------------------------------- #
MARKET_CFG = FeatureConfig(
    n_lags=2, roll_windows=(5, 21), drawdown_window=5, rvol_window=5, market_features=True
)


def test_market_features_add_the_documented_columns() -> None:
    columns = feature_columns(MARKET_CFG)
    for name in (
        "mkt_ret_lag_1",
        "mkt_ret_lag_5",
        "mkt_rvol_5",
        "rel_ret_lag_1",
        "abs_ret_lag_1",
        "vol_ratio",
    ):
        assert name in columns
        assert name in continuous_feature_columns(MARKET_CFG)
    assert "mkt_ret_lag_1" not in feature_columns(FeatureConfig())


def test_market_features_known_answer(price_panel: pd.DataFrame) -> None:
    dataset = build_dataset(price_panel, MARKET_CFG)
    returns = np.log(price_panel).diff()
    market = returns.mean(axis=1)
    date = dates_of(dataset)[40]
    previous = returns.index[returns.index.get_loc(date) - 1]
    row = dataset.loc[(date, "SYNA")]
    assert row["mkt_ret_lag_1"] == pytest.approx(market.loc[previous])
    assert row["rel_ret_lag_1"] == pytest.approx(
        returns.loc[previous, "SYNA"] - market.loc[previous]
    )
    assert row["abs_ret_lag_1"] == pytest.approx(abs(returns.loc[previous, "SYNA"]))


@pytest.mark.parametrize("cut_position", [120, 300])
def test_market_features_do_not_look_ahead(price_panel: pd.DataFrame, cut_position: int) -> None:
    cut = price_panel.index[cut_position]
    base = build_dataset(price_panel, MARKET_CFG)
    mutated = price_panel.copy()
    future = mutated.index > cut
    mutated.loc[future] = mutated.loc[future].to_numpy() * 1.7
    rebuilt = build_dataset(mutated, MARKET_CFG)
    pd.testing.assert_frame_equal(
        base[dates_of(base) <= cut], rebuilt[dates_of(rebuilt) <= cut], check_exact=True
    )
