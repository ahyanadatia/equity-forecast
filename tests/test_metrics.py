"""Tests for :mod:`equity_forecast.metrics`.

The Diebold-Mariano tests are deliberately paranoid: the headline claim of the
whole project ("model X beats the random walk") rests on that statistic, so the
known-answer tests recompute it from scratch with numpy/scipy primitives rather
than calling the implementation twice and comparing it with itself.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from equity_forecast.metrics import (
    DMResult,
    aggregate_metrics,
    diebold_mariano,
    directional_accuracy,
    dm_pvalue_matrix,
    hit_rate_vs_baseline,
    information_coefficient,
    mae,
    rmse,
)

# Two hand-written error series used by the known-answer DM tests.
E1 = np.array([0.10, -0.20, 0.35, 0.05, -0.40, 0.15, 0.25, -0.30, 0.02, 0.18])
E2 = np.array([0.30, -0.10, 0.20, 0.45, -0.20, 0.35, 0.05, -0.50, 0.22, 0.08])


def autocorrelated_errors(n: int = 60, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return two error series whose loss differential is strongly autocorrelated.

    The common AR(1) driver makes ``gamma_1`` and ``gamma_2`` of ``d`` large, so
    the ``h > 1`` long-run-variance terms genuinely bite.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=n)
    driver = np.empty(n)
    driver[0] = shocks[0]
    for t in range(1, n):
        driver[t] = 0.85 * driver[t - 1] + shocks[t]
    return 0.5 * driver, driver + 0.3


# --------------------------------------------------------------------------- #
# point metrics
# --------------------------------------------------------------------------- #
def test_rmse_and_mae_known_answer() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 4.0, 3.0, 0.0])
    # errors: 0, -2, 0, 4  ->  mean |e| = 6/4 = 1.5 ; mean e^2 = 20/4 = 5
    assert mae(truth, pred) == pytest.approx(1.5)
    assert rmse(truth, pred) == pytest.approx(math.sqrt(5.0))
    assert rmse(truth, truth) == 0.0
    assert mae(truth, truth) == 0.0


def test_metrics_accept_series_and_drop_nan_pairs() -> None:
    truth = pd.Series([1.0, 2.0, np.nan, 4.0])
    pred = pd.Series([1.0, 4.0, 3.0, np.nan])
    # Only rows 0 and 1 survive: errors 0 and -2.
    assert mae(truth, pred) == pytest.approx(1.0)
    assert rmse(truth, pred) == pytest.approx(math.sqrt(2.0))
    assert rmse(truth, pred) == rmse(truth.to_numpy(), pred.to_numpy())


def test_point_metrics_are_nan_when_nothing_survives() -> None:
    truth = np.array([1.0, np.nan])
    pred = np.array([np.nan, 2.0])
    assert math.isnan(rmse(truth, pred))
    assert math.isnan(mae(truth, pred))
    assert math.isnan(hit_rate_vs_baseline(truth, pred, pred))


def test_point_metrics_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        rmse(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="same length"):
        mae(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="same length"):
        hit_rate_vs_baseline(np.zeros(3), np.zeros(3), np.zeros(2))


def test_directional_accuracy_known_answer() -> None:
    truth = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    pred = np.array([0.20, -0.10, -0.30, 0.40, 0.01])
    # hits: +/+ , -/- , +/- , -/+ , +/+  -> 3 of 5
    assert directional_accuracy(truth, pred) == pytest.approx(3.0 / 5.0)


def test_directional_accuracy_ignores_zero_target_rows() -> None:
    truth = np.array([0.01, 0.0, -0.02, 0.0])
    pred = np.array([0.5, -0.5, -0.5, 0.5])
    # The two y_true == 0 rows drop out; both remaining rows are hits.
    assert directional_accuracy(truth, pred) == pytest.approx(1.0)

    # Adding more zero-target rows cannot move the score.
    padded_truth = np.concatenate([truth, np.zeros(50)])
    padded_pred = np.concatenate([pred, np.full(50, -1.0)])
    assert directional_accuracy(padded_truth, padded_pred) == pytest.approx(1.0)


def test_directional_accuracy_gives_zero_forecast_no_credit() -> None:
    """An all-zero forecast (the zero_return baseline) must score 0.0, not 0.5."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=200)
    assert directional_accuracy(truth, np.zeros(200)) == 0.0

    # A single exact-zero prediction among otherwise perfect ones is one miss.
    truth = np.array([0.01, -0.02, 0.03, -0.04])
    pred = np.array([0.01, -0.02, 0.0, -0.04])
    assert directional_accuracy(truth, pred) == pytest.approx(3.0 / 4.0)


def test_directional_accuracy_is_nan_without_directional_rows() -> None:
    assert math.isnan(directional_accuracy(np.zeros(5), np.ones(5)))
    assert math.isnan(directional_accuracy(np.array([np.nan, 0.0]), np.array([1.0, 1.0])))


def test_information_coefficient_is_one_for_monotone_transform() -> None:
    truth = np.array([0.10, -0.20, 0.35, 0.05, -0.40, 0.22, 0.01])
    assert information_coefficient(truth, np.exp(3.0 * truth)) == pytest.approx(1.0)
    assert information_coefficient(truth, -np.exp(3.0 * truth)) == pytest.approx(-1.0)


def test_information_coefficient_matches_scipy_spearman() -> None:
    rng = np.random.default_rng(11)
    truth = rng.normal(size=80)
    pred = 0.4 * truth + rng.normal(size=80)
    assert information_coefficient(truth, pred) == pytest.approx(
        float(stats.spearmanr(truth, pred).statistic)
    )


@pytest.mark.parametrize(
    ("truth", "pred"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0])),  # n < 3
        (np.ones(10), np.arange(10.0)),  # constant truth
        (np.arange(10.0), np.zeros(10)),  # constant forecast
    ],
)
def test_information_coefficient_is_nan_when_undefined(truth: np.ndarray, pred: np.ndarray) -> None:
    assert math.isnan(information_coefficient(truth, pred))


def test_hit_rate_vs_baseline_known_answer() -> None:
    truth = np.array([1.0, 1.0, 1.0, 1.0])
    pred = np.array([1.1, 0.0, 1.0, 3.0])
    base = np.array([1.5, 0.5, 1.0, 1.2])
    # |e_model| = .1, 1., 0., 2.  vs |e_base| = .5, .5, 0., .2
    # strictly smaller on rows 0 only -> 1/4 (row 2 is a tie, which does not count)
    assert hit_rate_vs_baseline(truth, pred, base) == pytest.approx(0.25)


def test_hit_rate_vs_baseline_bounds() -> None:
    truth = np.array([1.0, -1.0, 2.0])
    perfect = truth.copy()
    awful = truth + 10.0
    assert hit_rate_vs_baseline(truth, perfect, awful) == 1.0
    assert hit_rate_vs_baseline(truth, awful, perfect) == 0.0
    assert hit_rate_vs_baseline(truth, awful, awful) == 0.0


# --------------------------------------------------------------------------- #
# Diebold-Mariano: known answers computed by hand
# --------------------------------------------------------------------------- #
def test_diebold_mariano_known_answer_h1() -> None:
    """Recompute the h=1 statistic and p-value from first principles."""
    d = E1**2 - E2**2
    n = d.size
    dbar = float(np.mean(d))
    dev = d - dbar
    gamma_0 = float(np.mean(dev * dev))
    # h = 1 -> no autocovariance terms; lrv is just gamma_0.
    raw = dbar / math.sqrt(gamma_0 / n)
    hln = math.sqrt((n + 1 - 2 * 1 + 1 * (1 - 1) / n) / n)
    expected_stat = raw * hln
    expected_p = float(2.0 * stats.t.sf(abs(expected_stat), n - 1))

    # Guard the fixture itself: these are the numbers the assertions below rest on.
    assert n == 10
    assert dbar == pytest.approx(-0.02695, abs=1e-12)
    assert gamma_0 == pytest.approx(0.0103423225, abs=1e-12)
    assert expected_stat == pytest.approx(-0.7950070615029402, abs=1e-12)

    result = diebold_mariano(E1, E2)
    assert isinstance(result, DMResult)
    assert result.n_obs == n
    assert result.harvey_corrected is True
    assert result.statistic == pytest.approx(expected_stat, abs=1e-12)
    assert result.p_value == pytest.approx(expected_p, abs=1e-12)


def test_diebold_mariano_known_answer_h3_uses_autocovariance_lags() -> None:
    """The h=3 long-run variance must include 2*(gamma_1 + gamma_2)."""
    err1, err2 = autocorrelated_errors()
    d = err1**2 - err2**2
    n = d.size
    dbar = float(np.mean(d))
    dev = d - dbar
    gamma_0 = float(np.mean(dev * dev))
    gamma_1 = float(np.sum(dev[1:] * dev[:-1]) / n)
    gamma_2 = float(np.sum(dev[2:] * dev[:-2]) / n)
    lrv = gamma_0 + 2.0 * (gamma_1 + gamma_2)
    assert gamma_1 > 0.0  # the series really is autocorrelated
    assert lrv > 0.0

    h = 3
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    expected_stat = dbar / math.sqrt(lrv / n) * hln
    expected_p = float(2.0 * stats.t.sf(abs(expected_stat), n - 1))

    result = diebold_mariano(err1, err2, h=h)
    assert result.statistic == pytest.approx(expected_stat, abs=1e-12)
    assert result.p_value == pytest.approx(expected_p, abs=1e-12)


def test_diebold_mariano_known_answer_absolute_loss() -> None:
    d = np.abs(E1) - np.abs(E2)
    n = d.size
    dbar = float(np.mean(d))
    dev = d - dbar
    raw = dbar / math.sqrt(float(np.mean(dev * dev)) / n)
    expected = raw * math.sqrt((n - 1) / n)

    result = diebold_mariano(E1, E2, loss="abs")
    assert result.statistic == pytest.approx(expected, abs=1e-12)
    assert result.statistic != pytest.approx(diebold_mariano(E1, E2).statistic, abs=1e-6)


# --------------------------------------------------------------------------- #
# Diebold-Mariano: behaviour
# --------------------------------------------------------------------------- #
def test_diebold_mariano_sign_convention_model_one_better() -> None:
    """Model 1 with strictly smaller errors must give a negative, significant statistic."""
    rng = np.random.default_rng(0)
    err2 = rng.normal(size=200)
    err1 = 0.2 * err2
    assert np.all(np.abs(err1) <= np.abs(err2))

    better_first = diebold_mariano(err1, err2)
    assert better_first.statistic < 0.0
    assert better_first.p_value < 0.05

    worse_first = diebold_mariano(err2, err1)
    assert worse_first.statistic > 0.0
    assert worse_first.p_value < 0.05


def test_diebold_mariano_identical_errors_are_degenerate() -> None:
    rng = np.random.default_rng(5)
    errors = rng.normal(size=50)
    result = diebold_mariano(errors, errors.copy())
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.n_obs == 50
    assert result.harvey_corrected is False


def test_diebold_mariano_is_antisymmetric() -> None:
    err1, err2 = autocorrelated_errors()
    forward = diebold_mariano(err1, err2)
    reverse = diebold_mariano(err2, err1)
    assert forward.statistic == pytest.approx(-reverse.statistic, abs=1e-15)
    assert forward.p_value == pytest.approx(reverse.p_value, abs=1e-15)
    assert forward.n_obs == reverse.n_obs


def test_harvey_correction_shrinks_the_statistic() -> None:
    corrected = diebold_mariano(E1, E2, harvey_correction=True)
    uncorrected = diebold_mariano(E1, E2, harvey_correction=False)
    assert corrected.harvey_corrected is True
    assert uncorrected.harvey_corrected is False
    assert abs(corrected.statistic) < abs(uncorrected.statistic)
    assert corrected.p_value > uncorrected.p_value  # more conservative

    n = E1.size
    assert corrected.statistic == pytest.approx(
        uncorrected.statistic * math.sqrt((n - 1) / n), abs=1e-12
    )


def test_horizon_changes_the_statistic() -> None:
    err1, err2 = autocorrelated_errors()
    at_h1 = diebold_mariano(err1, err2, h=1).statistic
    at_h3 = diebold_mariano(err1, err2, h=3).statistic
    assert at_h1 != pytest.approx(at_h3, rel=1e-6)
    # Positive autocovariance inflates the long-run variance, so the statistic shrinks.
    assert abs(at_h3) < abs(at_h1)


def test_diebold_mariano_drops_nan_pairs() -> None:
    err1 = np.concatenate([E1, [np.nan, 0.5]])
    err2 = np.concatenate([E2, [0.5, np.nan]])
    result = diebold_mariano(err1, err2)
    clean = diebold_mariano(E1, E2)
    assert result.n_obs == E1.size
    assert result.statistic == pytest.approx(clean.statistic, abs=1e-15)


@pytest.mark.parametrize(
    ("kwargs", "err1", "err2", "match"),
    [
        ({}, np.zeros(5), np.zeros(4), "same length"),
        ({"loss": "huber"}, E1, E2, "unknown loss"),
        ({}, np.array([1.0]), np.array([2.0]), "at least 2"),
        ({}, np.array([np.nan, 1.0]), np.array([1.0, np.nan]), "at least 2"),
        ({"h": 0}, E1, E2, "h must be >= 1"),
    ],
)
def test_diebold_mariano_rejects_bad_input(
    kwargs: dict[str, object], err1: np.ndarray, err2: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        diebold_mariano(err1, err2, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# p-value matrix
# --------------------------------------------------------------------------- #
def test_dm_pvalue_matrix_is_symmetric_with_nan_diagonal() -> None:
    rng = np.random.default_rng(42)
    truth = rng.normal(size=120)
    errors = {
        "ridge": truth * 0.2 + rng.normal(scale=0.10, size=120),
        "arima": truth * 0.5 + rng.normal(scale=0.40, size=120),
        "zero_return": truth,
    }
    matrix = dm_pvalue_matrix(errors)

    assert list(matrix.index) == ["ridge", "arima", "zero_return"]
    assert list(matrix.columns) == list(matrix.index)
    assert bool(np.isnan(np.diag(matrix.to_numpy())).all())

    upper = matrix.to_numpy()
    assert upper == pytest.approx(upper.T, nan_ok=True)

    # Off-diagonal cells agree with a direct pairwise call.
    names = list(errors)
    for left in names:
        for right in names:
            if left == right:
                continue
            expected = diebold_mariano(errors[left], errors[right]).p_value
            assert float(matrix.loc[left, right]) == pytest.approx(expected, abs=1e-15)
            assert 0.0 <= float(matrix.loc[left, right]) <= 1.0


def test_dm_pvalue_matrix_respects_loss_and_horizon() -> None:
    err1, err2 = autocorrelated_errors()
    errors = {"a": err1, "b": err2}
    squared = dm_pvalue_matrix(errors)
    absolute = dm_pvalue_matrix(errors, loss="abs")
    lagged = dm_pvalue_matrix(errors, h=3)
    assert float(squared.loc["a", "b"]) != pytest.approx(float(absolute.loc["a", "b"]), rel=1e-6)
    assert float(squared.loc["a", "b"]) != pytest.approx(float(lagged.loc["a", "b"]), rel=1e-6)
    assert float(lagged.loc["a", "b"]) == pytest.approx(
        diebold_mariano(err1, err2, h=3).p_value, abs=1e-15
    )


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_metrics_produces_mean_and_std_columns() -> None:
    by_fold = pd.DataFrame(
        {
            "model": ["ridge", "ridge", "arima", "arima"],
            "fold_id": [0, 1, 0, 1],
            "rmse": [0.1, 0.3, 0.2, 0.4],
            "ic": [0.05, 0.15, -0.10, 0.10],
        }
    )
    out = aggregate_metrics(by_fold, ["model"], ["rmse", "ic"])

    assert list(out.columns) == ["model", "rmse_mean", "rmse_std", "ic_mean", "ic_std"]
    assert len(out) == 2

    row = out.set_index("model").loc["ridge"]
    assert float(row["rmse_mean"]) == pytest.approx(0.2)
    assert float(row["rmse_std"]) == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert float(row["ic_mean"]) == pytest.approx(0.10)
    assert float(row["ic_std"]) == pytest.approx(np.std([0.05, 0.15], ddof=1))


def test_aggregate_metrics_multi_key_and_single_fold_std() -> None:
    by_fold = pd.DataFrame(
        {
            "model": ["ridge", "ridge", "arima"],
            "ticker": ["AAA", "BBB", "AAA"],
            "rmse": [0.1, 0.2, 0.3],
        }
    )
    out = aggregate_metrics(by_fold, ["model", "ticker"], ["rmse"])
    assert list(out.columns) == ["model", "ticker", "rmse_mean", "rmse_std"]
    assert len(out) == 3
    # One observation per group: the mean exists, the dispersion honestly does not.
    assert out["rmse_mean"].notna().all()
    assert bool(out["rmse_std"].isna().all())


def test_aggregate_metrics_rejects_bad_columns() -> None:
    by_fold = pd.DataFrame({"model": ["a"], "rmse": [0.1]})
    with pytest.raises(KeyError, match="missing"):
        aggregate_metrics(by_fold, ["model"], ["nope"])
    with pytest.raises(KeyError, match="missing"):
        aggregate_metrics(by_fold, ["nope"], ["rmse"])
    with pytest.raises(ValueError, match="group_cols"):
        aggregate_metrics(by_fold, [], ["rmse"])
    with pytest.raises(ValueError, match="metric_cols"):
        aggregate_metrics(by_fold, ["model"], [])
