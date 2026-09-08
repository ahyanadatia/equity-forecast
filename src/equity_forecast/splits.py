"""Walk-forward (rolling-origin) cross-validation splits.

The evaluation protocol is a walk-forward one: every model is refitted on a
growing (``expanding``) or sliding (``rolling``) window of history and then
scored on the block of trading days that immediately follows it. An embargo of
``gap_days`` trading days is removed between the two so that a horizon-``h``
target realised on the last training day cannot overlap the first test day.

Geometry
--------
Given the dataset's sorted, unique trading calendar ``dates``:

* Training opens at ``dates[0]``. The first test block opens at the first
  trading day at least ``initial_train_years`` calendar years after ``dates[0]``
  (a :class:`pandas.DateOffset` in months, so fractional years work), pushed
  forward by a further ``gap_days`` trading days for the embargo.
* Each test block spans ``test_months`` calendar months of trading days. Blocks
  are anchored on absolute month offsets from the first test day, so they are
  contiguous, non-overlapping, strictly increasing and tile the whole
  post-initial-train calendar with no gaps and no month-arithmetic drift.
* ``train_end`` sits exactly ``gap_days`` trading days before ``test_start``: if
  ``test_start`` is calendar position ``i`` then ``train_end`` is
  ``dates[i - 1 - gap_days]``, so the number of trading days strictly between
  the two is ``gap_days``.
* ``expanding`` pins ``train_start`` at ``dates[0]``; ``rolling`` moves it to
  the first trading day within ``rolling_years`` of ``train_end``.

Trailing stub rule
------------------
The final block is only as long as whatever calendar is left. It is dropped when
it holds fewer than one third of the trading days a full ``test_months`` block
holds on this calendar, where "full" is estimated from the calendar's own
average number of trading days per month (so the rule makes no assumption about
the data frequency). This keeps the last fold from being a two-week stub whose
metrics are pure noise. A calendar that yields no surviving block raises
:class:`ValueError`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from equity_forecast.config import SplitConfig

__all__ = [
    "Fold",
    "describe_folds",
    "fold_masks",
    "generate_folds",
    "window_mask",
]

logger = logging.getLogger(__name__)

_MONTHS_PER_YEAR = 12
#: Mean length of a Gregorian month in days, used to convert a calendar span to months.
_DAYS_PER_MONTH = 30.436875
#: A trailing test block shorter than this fraction of a nominal block is discarded.
_STUB_FRACTION = 1.0 / 3.0


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: a train block, an embargo gap, then a test block.

    Attributes:
        fold_id: Zero-based position of the fold in the walk-forward sequence.
        train_start: First trading date of the training block (inclusive).
        train_end: Last trading date of the training block (inclusive).
        test_start: First trading date of the test block (inclusive).
        test_end: Last trading date of the test block (inclusive).
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _offset_months(years: float) -> int:
    """Convert a (possibly fractional) number of years to whole months.

    Args:
        years: Length of a period in calendar years.

    Returns:
        The nearest whole number of months.
    """
    return round(years * _MONTHS_PER_YEAR)


def _normalise_calendar(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return ``dates`` as a sorted, duplicate-free :class:`pandas.DatetimeIndex`.

    Args:
        dates: The dataset's trading calendar. Already sorted and unique in
            normal use; a stray duplicate or an unsorted input is repaired here
            rather than silently producing nonsense folds.

    Returns:
        The same dates, sorted ascending with duplicates removed.

    Raises:
        TypeError: If ``dates`` does not hold datetimes.
    """
    try:
        calendar = pd.DatetimeIndex(dates)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        msg = f"splits need a DatetimeIndex trading calendar, got {type(dates).__name__}"
        raise TypeError(msg) from exc
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        logger.debug("trading calendar was unsorted or contained duplicates; repairing")
        calendar = pd.DatetimeIndex(calendar.unique()).sort_values()
    return calendar


def _first_test_index(calendar: pd.DatetimeIndex, cfg: SplitConfig) -> int:
    """Return the calendar position where the first test block opens.

    Args:
        calendar: Sorted, unique trading calendar.
        cfg: Split geometry.

    Returns:
        The index into ``calendar`` of the first test day.

    Raises:
        ValueError: If the calendar cannot accommodate the initial training
            period plus the embargo and at least one test day.
    """
    anchor = calendar[0] + pd.DateOffset(months=_offset_months(cfg.initial_train_years))
    n_train = int(calendar.searchsorted(anchor, side="left"))
    first_test = n_train + cfg.gap_days
    if n_train < 1 or first_test >= len(calendar):
        msg = (
            f"trading calendar {calendar[0].date()}..{calendar[-1].date()} "
            f"({len(calendar)} days) is too short for initial_train_years="
            f"{cfg.initial_train_years} plus a {cfg.gap_days}-day embargo: "
            f"{n_train} training day(s) and no test day left over"
        )
        raise ValueError(msg)
    return first_test


def _test_blocks(
    calendar: pd.DatetimeIndex, first_test: int, test_months: int
) -> list[tuple[int, int]]:
    """Tile the calendar from ``first_test`` onwards into month-sized blocks.

    Args:
        calendar: Sorted, unique trading calendar.
        first_test: Index of the first test day.
        test_months: Nominal block length in calendar months.

    Returns:
        Inclusive ``(start, end)`` index pairs, contiguous and increasing.
    """
    n = len(calendar)
    base = calendar[first_test]
    blocks: list[tuple[int, int]] = []
    start = first_test
    step = 1
    while start < n:
        anchor = base + pd.DateOffset(months=test_months * step)
        end = min(int(calendar.searchsorted(anchor, side="left")) - 1, n - 1)
        end = max(end, start)
        blocks.append((start, end))
        start = end + 1
        step += 1
    return blocks


def _nominal_block_days(calendar: pd.DatetimeIndex, test_months: int) -> float:
    """Estimate how many trading days a full ``test_months`` block holds.

    The estimate comes from the calendar itself (its average number of entries
    per calendar month), so it is correct for daily, weekly or holiday-ridden
    calendars alike.

    Args:
        calendar: Sorted, unique trading calendar.
        test_months: Nominal block length in calendar months.

    Returns:
        The expected number of calendar entries in a full block.
    """
    span_days = (calendar[-1] - calendar[0]).days
    if span_days <= 0:
        return float(len(calendar))
    per_month = len(calendar) / (span_days / _DAYS_PER_MONTH)
    return per_month * test_months


def _drop_trailing_stub(
    calendar: pd.DatetimeIndex, blocks: list[tuple[int, int]], test_months: int
) -> list[tuple[int, int]]:
    """Drop the final block when the calendar ran out too early.

    Args:
        calendar: Sorted, unique trading calendar.
        blocks: Inclusive ``(start, end)`` index pairs from :func:`_test_blocks`.
        test_months: Nominal block length in calendar months.

    Returns:
        ``blocks`` without its last element if that element holds fewer than
        :data:`_STUB_FRACTION` of a nominal block's trading days.
    """
    if not blocks:
        return blocks
    start, end = blocks[-1]
    length = end - start + 1
    nominal = _nominal_block_days(calendar, test_months)
    if length < _STUB_FRACTION * nominal:
        logger.debug(
            "dropping trailing test block %s..%s (%d days < %.1f)",
            calendar[start].date(),
            calendar[end].date(),
            length,
            _STUB_FRACTION * nominal,
        )
        return blocks[:-1]
    return blocks


def _train_start_index(calendar: pd.DatetimeIndex, train_end: int, cfg: SplitConfig) -> int:
    """Return the calendar position where a fold's training block opens.

    Args:
        calendar: Sorted, unique trading calendar.
        train_end: Index of the fold's last training day.
        cfg: Split geometry.

    Returns:
        ``0`` in ``expanding`` mode; in ``rolling`` mode the first index within
        ``cfg.rolling_years`` of ``calendar[train_end]``.
    """
    if cfg.mode == "expanding":
        return 0
    cutoff = calendar[train_end] - pd.DateOffset(months=_offset_months(cfg.rolling_years))
    return int(calendar.searchsorted(cutoff, side="left"))


def generate_folds(dates: pd.DatetimeIndex, cfg: SplitConfig) -> list[Fold]:
    """Build the walk-forward folds for a trading calendar.

    Args:
        dates: The dataset's sorted, unique trading calendar.
        cfg: Split geometry (mode, initial train length, block length, embargo).

    Returns:
        Folds ordered by time, with contiguous non-overlapping test blocks and
        ``gap_days`` trading days embargoed between each ``train_end`` and its
        ``test_start``.

    Raises:
        ValueError: If the calendar is empty, or too short to yield a single
            fold whose test block survives the trailing-stub rule.
    """
    calendar = _normalise_calendar(dates)
    if len(calendar) == 0:
        msg = "cannot generate folds from an empty trading calendar"
        raise ValueError(msg)

    first_test = _first_test_index(calendar, cfg)
    blocks = _drop_trailing_stub(
        calendar, _test_blocks(calendar, first_test, cfg.test_months), cfg.test_months
    )
    if not blocks:
        msg = (
            f"trading calendar {calendar[0].date()}..{calendar[-1].date()} "
            f"({len(calendar)} days) leaves only a stub test block shorter than "
            f"{_STUB_FRACTION:.2f} of a {cfg.test_months}-month block; no fold can be built"
        )
        raise ValueError(msg)

    folds: list[Fold] = []
    for fold_id, (start, end) in enumerate(blocks):
        train_end = start - 1 - cfg.gap_days
        train_start = _train_start_index(calendar, train_end, cfg)
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=calendar[train_start],
                train_end=calendar[train_end],
                test_start=calendar[start],
                test_end=calendar[end],
            )
        )
    logger.debug(
        "built %d %s fold(s) over %s..%s (test_months=%d, gap_days=%d)",
        len(folds),
        cfg.mode,
        calendar[0].date(),
        calendar[-1].date(),
        cfg.test_months,
        cfg.gap_days,
    )
    return folds


def _date_level(index: pd.MultiIndex) -> pd.DatetimeIndex:
    """Extract the date level of a dataset index.

    Args:
        index: A ``(date, ticker)`` MultiIndex; the first level is used if the
            levels are unnamed.

    Returns:
        The date level as a :class:`pandas.DatetimeIndex`, one entry per row.

    Raises:
        TypeError: If that level does not hold datetimes.
    """
    level: int | str = "date" if "date" in list(index.names) else 0
    values = index.get_level_values(level)
    try:
        return pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        msg = f"index level {level!r} must hold datetimes, got dtype {values.dtype}"
        raise TypeError(msg) from exc


def fold_masks(
    index: pd.MultiIndex, fold: Fold
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Return boolean row masks selecting a fold's train and test rows.

    Args:
        index: The dataset's ``(date, ticker)`` MultiIndex.
        fold: The fold to select.

    Returns:
        ``(train_mask, test_mask)``, both aligned to ``index``. The two are
        disjoint by construction, since ``train_end < test_start``.
    """
    dates = _date_level(index)
    train = (dates >= fold.train_start) & (dates <= fold.train_end)
    test = (dates >= fold.test_start) & (dates <= fold.test_end)
    return np.asarray(train, dtype=bool), np.asarray(test, dtype=bool)


def window_mask(index: pd.MultiIndex, fold: Fold) -> npt.NDArray[np.bool_]:
    """Return the mask of a fold's full window, ``train_start..test_end``.

    This is the frame handed to :meth:`~equity_forecast.models.base.Forecaster.predict`:
    training rows, the embargoed gap rows and the test rows, so sequence and
    state-space models can warm up before the test block starts.

    Args:
        index: The dataset's ``(date, ticker)`` MultiIndex.
        fold: The fold to select.

    Returns:
        A boolean mask aligned to ``index`` covering ``train_start`` through
        ``test_end`` inclusive.
    """
    dates = _date_level(index)
    return np.asarray((dates >= fold.train_start) & (dates <= fold.test_end), dtype=bool)


def describe_folds(folds: list[Fold], *, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Summarise folds as one row per fold, for the run report.

    Args:
        folds: The folds to describe, as returned by :func:`generate_folds`.
        dates: Optional trading calendar. When given, ``n_train_days`` and
            ``n_test_days`` count *trading* days from that calendar; when
            omitted they count inclusive calendar days between the boundaries.

    Returns:
        A frame with columns ``fold_id, train_start, train_end, test_start,
        test_end, n_train_days, n_test_days``, in fold order.
    """
    calendar = None if dates is None else _normalise_calendar(dates)

    def _count(start: pd.Timestamp, end: pd.Timestamp) -> int:
        if calendar is None:
            return int((end - start).days) + 1
        return int(((calendar >= start) & (calendar <= end)).sum())

    rows = [
        {
            "fold_id": fold.fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "n_train_days": _count(fold.train_start, fold.train_end),
            "n_test_days": _count(fold.test_start, fold.test_end),
        }
        for fold in folds
    ]
    columns = [
        "fold_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "n_train_days",
        "n_test_days",
    ]
    if not rows:
        empty = pd.DataFrame({c: pd.Series(dtype="int64") for c in columns})
        for col in ("train_start", "train_end", "test_start", "test_end"):
            empty[col] = pd.Series(dtype="datetime64[ns]")
        return empty[columns]
    return pd.DataFrame(rows, columns=columns)
