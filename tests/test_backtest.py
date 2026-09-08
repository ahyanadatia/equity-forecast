"""Tests for :mod:`equity_forecast.backtest`.

The backtest is where a forecast turns into a claim about money, so these tests
lean on cases whose answer is known by construction rather than on anything
statistical: a signal of exactly zero must produce exactly zero pnl, perfect
foresight must earn exactly the sum of the absolute realised moves, and raising
costs must never raise the net Sharpe.

Note what is deliberately *absent*: there is no test asserting that a random or
arbitrary signal makes money. Such a test would either be a coin flip or, worse,
would be tuned until it passed - which is precisely the failure mode the whole
project exists to avoid.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from equity_forecast.backtest import (
    PORTFOLIO_COLUMNS,
    SUMMARY_KEYS,
    BacktestResult,
    build_positions,
    buy_and_hold,
    run_backtest,
    summarise_pnl,
)
from equity_forecast.config import BacktestConfig


def wide(columns: dict[str, Sequence[float]], start: str = "2021-01-04") -> pd.DataFrame:
    """Build a wide date x ticker panel from equal-length columns."""
    lengths = {len(v) for v in columns.values()}
    assert len(lengths) == 1, "all columns must have the same length"
    index = pd.bdate_range(start, periods=lengths.pop(), name="date")
    frame = pd.DataFrame(
        {name: np.asarray(values, dtype="float64") for name, values in columns.items()},
        index=index,
    )
    frame.columns.name = "ticker"
    return frame


def random_returns(n: int = 160, seed: int = 0, scale: float = 0.02) -> pd.DataFrame:
    """Return a two-ticker panel of pseudo-random daily returns."""
    rng = np.random.default_rng(seed)
    return wide({"AAA": rng.normal(0.0, scale, n), "BBB": rng.normal(0.0, scale, n)})


def flat_config(**overrides: float | None) -> BacktestConfig:
    """Return a :class:`BacktestConfig` with the given overrides applied."""
    base: dict[str, float | None] = {"cost_bps": 5.0, "threshold": 0.0}
    base.update(overrides)
    return BacktestConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# a signal of nothing must earn nothing
# --------------------------------------------------------------------------- #
def test_zero_signal_gives_a_flat_book() -> None:
    realised = random_returns()
    forecast = realised * 0.0

    result = run_backtest(forecast, realised, flat_config(cost_bps=5.0))

    assert isinstance(result, BacktestResult)
    np.testing.assert_array_equal(result.positions.to_numpy(), np.zeros(realised.shape))
    np.testing.assert_array_equal(result.gross_pnl.to_numpy(), np.zeros(realised.shape))
    np.testing.assert_array_equal(result.net_pnl.to_numpy(), np.zeros(realised.shape))
    np.testing.assert_array_equal(result.portfolio["turnover"].to_numpy(), np.zeros(len(realised)))
    assert result.summary["total_return"] == 0.0
    assert result.summary["avg_turnover"] == 0.0
    assert result.summary["pct_days_in_market"] == 0.0
    assert result.summary["sharpe"] == 0.0
    assert result.summary["max_drawdown"] == 0.0


def test_zero_signal_and_zero_cost_is_exactly_zero_pnl() -> None:
    realised = random_returns(seed=4)
    forecast = realised * 0.0

    result = run_backtest(forecast, realised, flat_config(cost_bps=0.0))

    for column in PORTFOLIO_COLUMNS:
        np.testing.assert_array_equal(result.portfolio[column].to_numpy(), np.zeros(len(realised)))
    assert result.summary["ann_return"] == 0.0
    assert result.summary["ann_vol"] == 0.0
    assert result.gross_summary["total_return"] == 0.0


# --------------------------------------------------------------------------- #
# perfect foresight: the upper bound on what any signal can earn
# --------------------------------------------------------------------------- #
def test_perfect_foresight_earns_the_sum_of_absolute_moves() -> None:
    realised = random_returns(n=200, seed=1)
    forecast = realised.copy()

    result = run_backtest(forecast, realised, flat_config(cost_bps=5.0))

    expected_gross = float(np.abs(realised.to_numpy()).sum())
    assert float(result.gross_pnl.to_numpy().sum()) == pytest.approx(expected_gross)
    assert expected_gross > 0.0
    assert result.gross_summary["total_return"] > 0.0
    # Net of 5 bps a perfect signal is still comfortably profitable.
    assert result.summary["total_return"] > 0.0
    assert result.summary["total_return"] < result.gross_summary["total_return"]
    assert result.summary["sharpe"] > 0.0


# --------------------------------------------------------------------------- #
# costs
# --------------------------------------------------------------------------- #
def test_net_sharpe_is_non_increasing_in_cost() -> None:
    n = 120
    rng = np.random.default_rng(3)
    realised = wide({"AAA": rng.normal(0.0, 0.015, n), "BBB": rng.normal(0.0, 0.015, n)})
    # Alternating "long / flat" so turnover is exactly 1.0 on every single day:
    # the cost is then a constant subtracted from each day's pnl, which leaves
    # the volatility untouched and makes the Sharpe ordering exact, not noisy.
    signal = np.where(np.arange(n) % 2 == 0, 0.01, 0.0)
    forecast = wide({"AAA": signal, "BBB": signal})

    sharpes: list[float] = []
    totals: list[float] = []
    vols: list[float] = []
    for cost_bps in (0.0, 1.0, 5.0, 20.0, 100.0):
        result = run_backtest(forecast, realised, flat_config(cost_bps=cost_bps))
        assert bool(result.portfolio["turnover"].eq(1.0).all())
        sharpes.append(result.summary["sharpe"])
        totals.append(result.summary["total_return"])
        vols.append(result.summary["ann_vol"])

    for before, after in pairwise(sharpes):
        assert after <= before + 1e-12
    for before, after in pairwise(totals):
        assert after < before
    for vol in vols[1:]:
        assert vol == pytest.approx(vols[0])


def test_portfolio_net_equals_gross_minus_cost() -> None:
    realised = random_returns(n=90, seed=5)
    rng = np.random.default_rng(9)
    forecast = wide(
        {"AAA": rng.normal(0.0, 0.01, 90), "BBB": rng.normal(0.0, 0.01, 90)},
    )
    vol_forecast = wide({"AAA": np.full(90, 0.012), "BBB": np.full(90, 0.008)})
    cfg = BacktestConfig(cost_bps=12.5, threshold=0.002, vol_target=0.1, vol_scale_cap=1.5)

    result = run_backtest(forecast, realised, cfg, vol_forecast=vol_forecast, hold_days=3)

    np.testing.assert_allclose(
        result.portfolio["net"].to_numpy(),
        result.portfolio["gross"].to_numpy() - result.portfolio["cost"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.portfolio["gross"].to_numpy(),
        result.gross_pnl.mean(axis=1).to_numpy(),
        rtol=0.0,
        atol=1e-15,
    )
    expected_cost = (
        result.positions.sub(result.positions.shift(1, fill_value=0.0)).abs() * 12.5 / 1e4
    )
    np.testing.assert_allclose(
        result.net_pnl.to_numpy(),
        result.gross_pnl.to_numpy() - expected_cost.to_numpy(),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.portfolio["cost"].to_numpy(),
        expected_cost.mean(axis=1).to_numpy(),
        rtol=0.0,
        atol=1e-15,
    )


# --------------------------------------------------------------------------- #
# turnover
# --------------------------------------------------------------------------- #
def test_a_constant_position_is_paid_for_once() -> None:
    realised = random_returns(n=40, seed=6)
    forecast = wide({"AAA": np.ones(40), "BBB": np.ones(40)})

    result = run_backtest(forecast, realised, flat_config(cost_bps=10.0))

    turnover = result.portfolio["turnover"]
    assert turnover.iloc[0] == 1.0
    np.testing.assert_array_equal(turnover.to_numpy()[1:], np.zeros(39))
    assert result.summary["avg_turnover"] == pytest.approx(1.0 / 40.0)
    assert result.summary["pct_days_in_market"] == 1.0
    # Only the entry is charged, at 10 bps on a unit position.
    assert float(result.portfolio["cost"].sum()) == pytest.approx(10.0 / 1e4)


def test_an_alternating_signal_pays_full_turnover_every_day() -> None:
    n = 30
    alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    realised = random_returns(n=n, seed=7)
    forecast = wide({"AAA": alternating, "BBB": alternating})

    result = run_backtest(forecast, realised, flat_config(cost_bps=1.0))

    turnover = result.portfolio["turnover"]
    assert turnover.iloc[0] == 1.0
    np.testing.assert_array_equal(turnover.to_numpy()[1:], np.full(n - 1, 2.0))
    np.testing.assert_allclose(result.portfolio["cost"].to_numpy(), turnover.to_numpy() * 1.0 / 1e4)


# --------------------------------------------------------------------------- #
# threshold
# --------------------------------------------------------------------------- #
def test_raising_the_threshold_weakly_reduces_time_in_market() -> None:
    rng = np.random.default_rng(11)
    n = 150
    realised = random_returns(n=n, seed=12)
    forecast = wide({"AAA": rng.normal(0.0, 0.01, n), "BBB": rng.normal(0.0, 0.01, n)})

    occupancy = [
        run_backtest(forecast, realised, flat_config(threshold=t)).summary["pct_days_in_market"]
        for t in (0.0, 0.002, 0.005, 0.01, 0.05)
    ]

    assert occupancy[0] > 0.0
    for before, after in pairwise(occupancy):
        assert after <= before + 1e-15
    assert occupancy[-1] == 0.0


def test_threshold_above_every_forecast_gives_a_flat_book() -> None:
    realised = random_returns(n=50, seed=13)
    forecast = wide({"AAA": np.full(50, 0.001), "BBB": np.full(50, -0.001)})
    above_everything = 0.01

    result = run_backtest(forecast, realised, flat_config(threshold=above_everything))

    np.testing.assert_array_equal(result.positions.to_numpy(), np.zeros((50, 2)))
    assert result.summary["total_return"] == 0.0
    assert result.summary["avg_turnover"] == 0.0
    assert result.summary["pct_days_in_market"] == 0.0


# --------------------------------------------------------------------------- #
# volatility targeting
# --------------------------------------------------------------------------- #
def test_vol_targeting_scales_positions_and_respects_the_cap() -> None:
    n = 6
    forecast = wide({"AAA": np.ones(n), "BBB": -np.ones(n)})
    # AAA: 1% daily vol -> 15.87% annualised -> scale 0.63 (uncapped).
    # BBB: 0.1% daily vol -> 1.59% annualised -> scale 6.3, clipped to the cap.
    vol_forecast = wide({"AAA": np.full(n, 0.01), "BBB": np.full(n, 0.001)})

    positions = build_positions(
        forecast, vol_forecast=vol_forecast, vol_target=0.10, cap=2.0, ann_factor=252
    )

    expected_aaa = 0.10 / (0.01 * np.sqrt(252.0))
    assert expected_aaa < 2.0
    np.testing.assert_allclose(positions["AAA"].to_numpy(), np.full(n, expected_aaa))
    np.testing.assert_allclose(positions["BBB"].to_numpy(), np.full(n, -2.0))


def test_vol_targeting_falls_back_when_vol_is_missing_or_zero() -> None:
    forecast = wide({"AAA": [1.0, 1.0, -1.0, 1.0]})
    vol_forecast = wide({"AAA": [0.01, 0.0, np.nan, -0.01]})

    positions = build_positions(
        forecast, vol_forecast=vol_forecast, vol_target=0.10, cap=2.0, ann_factor=252
    )

    scaled = 0.10 / (0.01 * np.sqrt(252.0))
    np.testing.assert_allclose(positions["AAA"].to_numpy(), [scaled, 1.0, -1.0, 1.0])
    assert np.isfinite(positions.to_numpy()).all()


def test_vol_targeting_needs_both_inputs() -> None:
    forecast = wide({"AAA": [1.0, -1.0]})
    vol_forecast = wide({"AAA": [0.01, 0.01]})

    only_vol = build_positions(forecast, vol_forecast=vol_forecast)
    only_target = build_positions(forecast, vol_target=0.10)

    np.testing.assert_array_equal(only_vol.to_numpy(), [[1.0], [-1.0]])
    np.testing.assert_array_equal(only_target.to_numpy(), [[1.0], [-1.0]])


# --------------------------------------------------------------------------- #
# holding period
# --------------------------------------------------------------------------- #
def test_hold_days_freezes_the_book_and_cuts_turnover() -> None:
    n = 20
    hold = 5
    alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    forecast = wide({"AAA": alternating, "BBB": alternating})
    realised = random_returns(n=n, seed=14)

    held = build_positions(forecast, hold_days=hold)
    values = held.to_numpy()
    for start in range(0, n, hold):
        block = values[start : start + hold]
        np.testing.assert_array_equal(block, np.repeat(block[:1], len(block), axis=0))
    # Refresh rows 0, 5, 10, 15 of an alternating signal give +1, -1, +1, -1.
    np.testing.assert_array_equal(values[::hold, 0], [1.0, -1.0, 1.0, -1.0])

    daily = run_backtest(forecast, realised, flat_config(), hold_days=1)
    weekly = run_backtest(forecast, realised, flat_config(), hold_days=hold)
    assert daily.summary["avg_turnover"] == pytest.approx((1.0 + 2.0 * (n - 1)) / n)
    assert weekly.summary["avg_turnover"] == pytest.approx((1.0 + 2.0 * 3.0) / n)
    assert weekly.summary["avg_turnover"] < daily.summary["avg_turnover"]


# --------------------------------------------------------------------------- #
# summary statistics
# --------------------------------------------------------------------------- #
def test_max_drawdown_is_a_positive_number_on_the_cumulative_sum() -> None:
    # cum = [0.10, -0.20, -0.15, -0.25, 0.15]; peak (starting at 0) = [0.10, ...],
    # so the worst peak-to-trough fall is 0.10 -> -0.25 = 0.35.
    pnl = pd.Series([0.10, -0.30, 0.05, -0.10, 0.40])
    assert summarise_pnl(pnl)["max_drawdown"] == pytest.approx(0.35)

    # A loss on day one is a drawdown from the initial zero level, not free.
    assert summarise_pnl(pd.Series([-0.20, 0.10]))["max_drawdown"] == pytest.approx(0.20)

    # A series that only ever gains has no drawdown at all.
    assert summarise_pnl(pd.Series([0.01, 0.02, 0.03]))["max_drawdown"] == 0.0


def test_summarise_pnl_keys_and_known_values() -> None:
    pnl = pd.Series([0.01, -0.005, 0.02, 0.0, -0.001])
    turnover = pd.Series([1.0, 0.0, 2.0, 0.0, 1.0])
    positions = pd.Series([1.0, 1.0, -1.0, 0.0, 0.0])

    stats = summarise_pnl(pnl, turnover=turnover, positions=positions, ann_factor=252)

    assert tuple(stats) == SUMMARY_KEYS
    expected_ann_return = float(pnl.mean()) * 252
    expected_ann_vol = float(pnl.std(ddof=1)) * np.sqrt(252)
    expected_downside = float(np.sqrt(np.mean(np.minimum(pnl.to_numpy(), 0.0) ** 2))) * np.sqrt(252)
    assert stats["ann_return"] == pytest.approx(expected_ann_return)
    assert stats["ann_vol"] == pytest.approx(expected_ann_vol)
    assert stats["sharpe"] == pytest.approx(expected_ann_return / expected_ann_vol)
    assert stats["sortino"] == pytest.approx(expected_ann_return / expected_downside)
    assert stats["avg_turnover"] == pytest.approx(0.8)
    assert stats["pct_days_in_market"] == pytest.approx(0.6)
    assert stats["total_return"] == pytest.approx(float(pnl.sum()))
    assert stats["n_days"] == 5.0


def test_summarise_pnl_degenerate_cases_are_finite() -> None:
    constant = summarise_pnl(pd.Series([0.01, 0.01, 0.01]))
    assert constant["ann_vol"] == 0.0
    assert constant["sharpe"] == 0.0
    assert constant["sortino"] == 0.0  # no downside -> 0.0, never inf

    empty = summarise_pnl(pd.Series([], dtype="float64"))
    assert empty == dict.fromkeys(SUMMARY_KEYS, 0.0)

    single = summarise_pnl(pd.Series([0.02]))
    assert single["ann_vol"] == 0.0
    assert single["n_days"] == 1.0
    assert all(np.isfinite(v) for v in single.values())


# --------------------------------------------------------------------------- #
# result plumbing
# --------------------------------------------------------------------------- #
def test_per_ticker_and_gross_summary_are_populated() -> None:
    realised = random_returns(n=80, seed=15)
    forecast = realised.copy()

    result = run_backtest(forecast, realised, flat_config(cost_bps=5.0))

    assert result.per_ticker.index.name == "ticker"
    assert list(result.per_ticker.index) == ["AAA", "BBB"]
    assert tuple(result.per_ticker.columns) == SUMMARY_KEYS
    for ticker in ("AAA", "BBB"):
        expected = summarise_pnl(
            result.net_pnl[ticker],
            turnover=result.positions[ticker]
            .sub(result.positions[ticker].shift(1, fill_value=0.0))
            .abs(),
            positions=result.positions[ticker],
        )
        for key, value in expected.items():
            assert result.per_ticker.loc[ticker, key] == pytest.approx(value)

    assert tuple(result.gross_summary) == SUMMARY_KEYS
    assert result.gross_summary["total_return"] > result.summary["total_return"]
    assert result.gross_summary["avg_turnover"] == result.summary["avg_turnover"]


def test_backtest_uses_only_the_common_dates_and_tickers() -> None:
    realised = random_returns(n=30, seed=16)
    forecast = realised.copy()
    forecast["CCC"] = 0.01
    extra_day = pd.DataFrame(
        [[0.01, 0.01, 0.01]],
        index=pd.DatetimeIndex(["2019-01-02"], name="date"),
        columns=forecast.columns,
    )
    forecast = pd.concat([extra_day, forecast])

    result = run_backtest(forecast, realised, flat_config())

    assert list(result.positions.columns) == ["AAA", "BBB"]
    pd.testing.assert_index_equal(result.positions.index, realised.index)
    assert result.summary["n_days"] == 30.0

    disjoint = realised.rename(columns={"AAA": "ZZZ", "BBB": "YYY"})
    with pytest.raises(ValueError, match="no common"):
        run_backtest(disjoint, realised, flat_config())


def test_build_positions_rejects_impossible_arguments() -> None:
    forecast = wide({"AAA": [1.0, -1.0]})
    with pytest.raises(ValueError, match="hold_days"):
        build_positions(forecast, hold_days=0)
    with pytest.raises(ValueError, match="threshold"):
        build_positions(forecast, threshold=-0.1)
    with pytest.raises(ValueError, match="cap"):
        build_positions(forecast, cap=-1.0)
    with pytest.raises(ValueError, match="ann_factor"):
        build_positions(forecast, ann_factor=0)
    with pytest.raises(ValueError, match="ann_factor"):
        summarise_pnl(pd.Series([0.01]), ann_factor=-1)


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #
def test_buy_and_hold_is_the_equal_weight_long_only_book() -> None:
    realised = random_returns(n=45, seed=17)

    benchmark = buy_and_hold(realised)

    assert tuple(benchmark.columns) == PORTFOLIO_COLUMNS
    np.testing.assert_allclose(benchmark["gross"].to_numpy(), realised.mean(axis=1).to_numpy())
    np.testing.assert_array_equal(benchmark["net"].to_numpy(), benchmark["gross"].to_numpy())
    np.testing.assert_array_equal(benchmark["cost"].to_numpy(), np.zeros(45))
    assert benchmark["turnover"].iloc[0] == 1.0
    np.testing.assert_array_equal(benchmark["turnover"].to_numpy()[1:], np.zeros(44))

    # It is exactly what an always-long, cost-free backtest produces.
    always_long = run_backtest(realised * 0.0 + 1.0, realised, flat_config(cost_bps=0.0)).portfolio
    np.testing.assert_allclose(
        benchmark["gross"].to_numpy(), always_long["gross"].to_numpy(), atol=1e-15
    )
