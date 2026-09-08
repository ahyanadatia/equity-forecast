"""Tests for the walk-forward split geometry.

The critical property here is that no fold can see its own future: the training
block always ends strictly before the test block starts, with exactly
``gap_days`` trading days embargoed in between.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from equity_forecast.config import SplitConfig
from equity_forecast.splits import Fold, describe_folds, fold_masks, generate_folds, window_mask

# A synthetic 2010-2026 daily trading calendar. No data and no network needed.
LONG_CALENDAR = pd.bdate_range("2010-01-04", "2026-01-01", name="date")


def _positions(calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    return {ts: i for i, ts in enumerate(calendar)}


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
@pytest.mark.parametrize("gap_days", [0, 1, 5])
def test_no_leakage_and_embargo_is_exact(mode: str, gap_days: int) -> None:
    """THE critical one: train ends before test starts, with an exact embargo."""
    cfg = SplitConfig(
        mode=mode,
        initial_train_years=5.0,
        test_months=6,
        rolling_years=3.0,
        gap_days=gap_days,
    )
    calendar = LONG_CALENDAR
    folds = generate_folds(calendar, cfg)
    assert folds

    pos = _positions(calendar)
    for fold in folds:
        assert fold.train_start <= fold.train_end
        assert fold.test_start <= fold.test_end

        train_dates = calendar[(calendar >= fold.train_start) & (calendar <= fold.train_end)]
        test_dates = calendar[(calendar >= fold.test_start) & (calendar <= fold.test_end)]
        assert len(train_dates) > 0
        assert len(test_dates) > 0

        # No leakage: every training date precedes every test date.
        assert train_dates.max() < test_dates.min()

        # The embargo is exactly gap_days *trading* days wide.
        embargoed = calendar[(calendar > fold.train_end) & (calendar < fold.test_start)]
        assert len(embargoed) == gap_days
        assert pos[fold.test_start] - pos[fold.train_end] - 1 == gap_days
        # ...which is the same statement as train_end == dates[i - 1 - gap_days].
        assert fold.train_end == calendar[pos[fold.test_start] - 1 - gap_days]


def test_test_blocks_are_contiguous_non_overlapping_and_increasing() -> None:
    """Consecutive test blocks abut exactly, never overlap, and move forward."""
    cfg = SplitConfig(initial_train_years=4.0, test_months=3, gap_days=2)
    calendar = LONG_CALENDAR
    folds = generate_folds(calendar, cfg)
    pos = _positions(calendar)

    assert len(folds) > 1
    assert [f.fold_id for f in folds] == list(range(len(folds)))
    for previous, current in itertools.pairwise(folds):
        assert previous.test_end < current.test_start
        # Contiguous: the next block opens on the very next trading day.
        assert pos[current.test_start] == pos[previous.test_end] + 1
        assert current.test_start > previous.test_start
        assert previous.train_end < current.train_end


def test_test_blocks_tile_the_post_initial_train_calendar() -> None:
    """The union of the test blocks covers the post-train calendar with no gaps."""
    cfg = SplitConfig(initial_train_years=5.0, test_months=6, gap_days=1)
    calendar = LONG_CALENDAR
    folds = generate_folds(calendar, cfg)
    pos = _positions(calendar)

    covered: list[int] = []
    for fold in folds:
        covered.extend(range(pos[fold.test_start], pos[fold.test_end] + 1))
    first, last = pos[folds[0].test_start], pos[folds[-1].test_end]
    assert covered == list(range(first, last + 1))
    assert len(covered) == len(set(covered))

    # The first test day is the first date >= start + 5y, pushed on by the embargo.
    anchor = int(calendar.searchsorted(calendar[0] + pd.DateOffset(years=5), side="left"))
    assert first == anchor + cfg.gap_days

    # Anything left unclaimed at the end is a dropped stub: under a third of a block.
    block_lengths = [pos[f.test_end] - pos[f.test_start] + 1 for f in folds]
    leftover = len(calendar) - last - 1
    assert leftover * 3 < float(np.median(block_lengths))


def test_expanding_mode_keeps_train_start_fixed_and_grows_nested_windows() -> None:
    """Expanding training sets all start at dates[0] and strictly nest."""
    cfg = SplitConfig(mode="expanding", initial_train_years=5.0, test_months=6, gap_days=1)
    calendar = LONG_CALENDAR
    folds = generate_folds(calendar, cfg)

    assert len({f.train_start for f in folds}) == 1
    assert folds[0].train_start == calendar[0]

    sizes = []
    previous: set[pd.Timestamp] = set()
    for fold in folds:
        train = set(calendar[(calendar >= fold.train_start) & (calendar <= fold.train_end)])
        assert previous <= train  # nested
        assert len(train) > len(previous)  # growing
        previous = train
        sizes.append(len(train))
    assert sizes == sorted(sizes)


def test_rolling_mode_window_is_bounded_and_roughly_constant() -> None:
    """Rolling training windows slide, stay within rolling_years, and keep their size."""
    rolling_years = 3.0
    cfg = SplitConfig(
        mode="rolling",
        initial_train_years=5.0,
        test_months=6,
        rolling_years=rolling_years,
        gap_days=1,
    )
    calendar = LONG_CALENDAR
    folds = generate_folds(calendar, cfg)
    pos = _positions(calendar)

    lengths = []
    for fold in folds:
        cutoff = fold.train_end - pd.DateOffset(months=36)
        # Bounded: never reaches back further than rolling_years before train_end.
        assert fold.train_start >= cutoff
        # Tight: the previous trading day would have breached the bound.
        assert calendar[pos[fold.train_start] - 1] < cutoff
        lengths.append(pos[fold.train_end] - pos[fold.train_start] + 1)

    nominal = rolling_years * 252
    assert max(lengths) - min(lengths) <= 15
    assert all(0.9 * nominal <= n <= 1.1 * nominal for n in lengths)
    # The window really slides rather than expanding.
    assert folds[-1].train_start > folds[0].train_start


def test_default_config_over_2010_2026_gives_at_least_15_folds() -> None:
    """FR-S2: the shipped default geometry yields a long walk-forward sequence."""
    folds = generate_folds(LONG_CALENDAR, SplitConfig())
    assert len(folds) >= 15
    assert folds[0].train_start == LONG_CALENDAR[0]
    assert folds[-1].test_end <= LONG_CALENDAR[-1]


def test_fold_masks_and_window_mask_are_consistent_on_a_multiindex() -> None:
    """Masks select the right rows and nest: train | test <= window, train & test empty."""
    calendar = pd.bdate_range("2016-01-04", "2021-12-31", name="date")
    tickers = ["AAA", "BBB", "CCC"]
    index = pd.MultiIndex.from_product([calendar, tickers], names=["date", "ticker"])
    cfg = SplitConfig(initial_train_years=2.0, test_months=6, gap_days=3)
    folds = generate_folds(calendar, cfg)
    assert folds

    row_dates = index.get_level_values("date")
    for fold in folds:
        train, test = fold_masks(index, fold)
        window = window_mask(index, fold)

        for mask in (train, test, window):
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.bool_
            assert mask.shape == (len(index),)

        assert not (train & test).any()
        assert not (train & ~window).any()
        assert not (test & ~window).any()

        # Selected rows are exactly the right dates, all tickers present.
        assert set(row_dates[train]) == set(
            calendar[(calendar >= fold.train_start) & (calendar <= fold.train_end)]
        )
        assert set(row_dates[test]) == set(
            calendar[(calendar >= fold.test_start) & (calendar <= fold.test_end)]
        )
        assert train.sum() % len(tickers) == 0
        assert row_dates[test].min() == fold.test_start
        assert row_dates[test].max() == fold.test_end

        # The window is train + embargo + test, and nothing else.
        assert window.sum() == train.sum() + test.sum() + cfg.gap_days * len(tickers)


@pytest.mark.parametrize(
    ("calendar", "cfg"),
    [
        (pd.DatetimeIndex([]), SplitConfig()),
        (pd.bdate_range("2020-01-01", "2021-01-01"), SplitConfig()),
        (pd.bdate_range("2010-01-04", "2015-01-20"), SplitConfig()),
        (pd.bdate_range("2010-01-04", "2020-01-01"), SplitConfig(initial_train_years=0.0)),
    ],
    ids=["empty", "shorter-than-initial-train", "stub-only-test-block", "no-training-history"],
)
def test_too_short_calendar_raises_value_error(
    calendar: pd.DatetimeIndex, cfg: SplitConfig
) -> None:
    """A calendar that cannot yield a usable fold fails loudly."""
    with pytest.raises(ValueError, match="calendar"):
        generate_folds(calendar, cfg)


def test_trailing_stub_is_dropped_but_a_substantial_partial_block_is_kept() -> None:
    """A tail under a third of a block is discarded; a longer tail becomes a fold."""
    cfg = SplitConfig(initial_train_years=5.0, test_months=6, gap_days=1)

    stub = generate_folds(pd.bdate_range("2010-01-04", "2016-01-20"), cfg)
    assert len(stub) == 2
    assert stub[-1].test_end == pd.Timestamp("2016-01-05")

    kept = generate_folds(pd.bdate_range("2010-01-04", "2016-05-31"), cfg)
    assert len(kept) == 3
    assert kept[-1].test_start == pd.Timestamp("2016-01-06")
    assert kept[-1].test_end == pd.Timestamp("2016-05-31")


def test_describe_folds_reports_boundaries_and_day_counts() -> None:
    """One row per fold, the four boundaries, and train/test day counts."""
    calendar = LONG_CALENDAR
    folds = generate_folds(calendar, SplitConfig())
    table = describe_folds(folds, dates=calendar)

    assert list(table.columns) == [
        "fold_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "n_train_days",
        "n_test_days",
    ]
    assert len(table) == len(folds)
    assert table["fold_id"].tolist() == [f.fold_id for f in folds]
    assert table["test_start"].tolist() == [f.test_start for f in folds]

    pos = _positions(calendar)
    expected_test = [pos[f.test_end] - pos[f.test_start] + 1 for f in folds]
    assert table["n_test_days"].tolist() == expected_test
    assert table["n_train_days"].tolist() == [
        pos[f.train_end] - pos[f.train_start] + 1 for f in folds
    ]
    assert table["n_train_days"].is_monotonic_increasing  # expanding

    # Without a calendar the counts fall back to inclusive calendar days.
    plain = describe_folds(folds)
    assert plain["n_test_days"].tolist() == [(f.test_end - f.test_start).days + 1 for f in folds]
    assert describe_folds([]).empty


def test_unsorted_or_duplicated_calendar_is_normalised() -> None:
    """A shuffled calendar with duplicates yields the same folds as a clean one."""
    calendar = pd.bdate_range("2010-01-04", "2020-01-01", name="date")
    messy = calendar.append(calendar[:50])[::-1]
    cfg = SplitConfig(initial_train_years=5.0, test_months=6, gap_days=1)
    assert generate_folds(messy, cfg) == generate_folds(calendar, cfg)


def test_fractional_initial_train_years_uses_a_month_offset() -> None:
    """1.5 years means 18 calendar months, not 1 or 2 years."""
    calendar = pd.bdate_range("2015-01-01", "2020-01-01", name="date")
    cfg = SplitConfig(initial_train_years=1.5, test_months=3, gap_days=0)
    folds = generate_folds(calendar, cfg)
    assert folds[0].test_start == calendar[int(calendar.searchsorted("2016-07-01"))]
    assert folds[0].train_end < folds[0].test_start


def test_fold_is_a_frozen_value_object() -> None:
    """Folds are hashable, comparable and immutable."""
    fold = Fold(
        fold_id=0,
        train_start=pd.Timestamp("2015-01-02"),
        train_end=pd.Timestamp("2019-12-31"),
        test_start=pd.Timestamp("2020-01-02"),
        test_end=pd.Timestamp("2020-06-30"),
    )
    assert fold == Fold(0, *[fold.train_start, fold.train_end, fold.test_start, fold.test_end])
    with pytest.raises(AttributeError):
        fold.fold_id = 1  # type: ignore[misc]
