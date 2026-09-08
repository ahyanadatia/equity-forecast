"""Tests for price caching and calendar alignment.

Every test here is offline. The network-facing half of :mod:`equity_forecast.data`
is exercised by monkeypatching ``yfinance.download``; the alignment half is
exercised through :func:`~equity_forecast.data.align_calendar`, which is a pure
function of hand-built frames.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yfinance

from equity_forecast.config import DataConfig
from equity_forecast.data import align_calendar, download_prices, load_prices

FIXTURE_TICKERS = ("SYNA", "SYNB", "SYNC")

# The holes scripts/make_fixtures.py punches into SYNC, as (gap start, gap length).
SYNC_GAPS: tuple[tuple[str, int], ...] = (
    ("2015-02-23", 2),  # Mon + Tue
    ("2015-07-30", 1),  # a lone Thursday
    ("2016-07-14", 3),  # Thu + Fri + Mon
)
SYNC_MISSING = [
    pd.Timestamp(d)
    for d in ("2015-02-23", "2015-02-24", "2015-07-30", "2016-07-14", "2016-07-15", "2016-07-18")
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_cache(tmp_path: Path, fixture_dir: Path, tickers: tuple[str, ...]) -> Path:
    """Copy the requested fixture parquets into a fresh cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    for ticker in tickers:
        shutil.copy(fixture_dir / f"{ticker}.parquet", cache / f"{ticker}.parquet")
    return cache


def ohlcv(index: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    """Build a minimal, well-formed OHLCV frame over ``index``."""
    close = pd.Series(base + np.arange(len(index), dtype="float64"), index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(index), 1_000, dtype="int64"),
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


def offline_cfg(
    cache: Path, tickers: tuple[str, ...] = FIXTURE_TICKERS, **kwargs: Any
) -> DataConfig:
    """Build an offline :class:`DataConfig` pointed at ``cache``."""
    params: dict[str, Any] = {
        "tickers": tickers,
        "start": "2015-01-01",
        "end": "2017-12-31",
        "cache_dir": str(cache),
        "offline": True,
        "max_missing_frac": 0.02,
        "max_ffill_days": 1,
    }
    params.update(kwargs)
    return DataConfig(**params)


def fake_yf_download(
    tickers: tuple[str, ...],
    n_rows: int = 6,
    group_by: str = "ticker",
    flat: bool = False,
) -> pd.DataFrame:
    """Build a frame shaped like ``yfinance.download`` output.

    Args:
        tickers: Tickers the frame should carry.
        n_rows: Number of business days.
        group_by: ``"ticker"`` puts tickers on level 0, ``"column"`` on level 1.
        flat: If ``True``, return the flat single-ticker shape instead.

    Returns:
        A frame with yfinance's capitalised field names.
    """
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n_rows), name="Date")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    blocks: dict[tuple[str, str] | str, Any] = {}
    for t_i, ticker in enumerate(tickers):
        close = 1000.0 + 100.0 * t_i + np.arange(n_rows, dtype="float64")
        values = {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n_rows, 5_000, dtype="int64"),
        }
        for field in fields:
            if flat:
                blocks[field] = values[field]
            elif group_by == "ticker":
                blocks[(ticker, field)] = values[field]
            else:
                blocks[(field, ticker)] = values[field]
    frame = pd.DataFrame(blocks, index=dates)
    if not flat:
        frame.columns = pd.MultiIndex.from_tuples(list(blocks))
    return frame


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace ``yfinance.download`` with a tripwire that fails the test if called."""
    calls: list[Any] = []

    def boom(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append((args, kwargs))
        msg = "yfinance.download was called, but this code path must never touch the network"
        raise AssertionError(msg)

    monkeypatch.setattr(yfinance, "download", boom)
    return calls


# --------------------------------------------------------------------------- #
# 1. schema of the loaded panel
# --------------------------------------------------------------------------- #
def test_load_prices_offline_returns_contract_shaped_panel(
    tmp_path: Path,
    fixture_dir: Path,
    raw_fixture_frames: dict[str, pd.DataFrame],
    no_network: list[Any],
) -> None:
    cache = make_cache(tmp_path, fixture_dir, FIXTURE_TICKERS)
    panel = load_prices(offline_cfg(cache))

    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.name == "date"
    assert panel.index.tz is None
    assert panel.index.is_monotonic_increasing
    assert panel.index.is_unique

    assert panel.columns.name == "ticker"
    assert list(panel.columns) == list(FIXTURE_TICKERS)
    assert (panel.dtypes == np.dtype("float64")).all()

    # Union calendar: SYNA/SYNB have every business day, SYNC is a subset of it.
    union = raw_fixture_frames["SYNA"].index.union(raw_fixture_frames["SYNC"].index)
    assert len(panel) == len(union)
    assert panel.shape == (len(union), 3)
    assert not no_network


def test_panel_values_are_the_fixture_adjusted_closes(
    tmp_path: Path,
    fixture_dir: Path,
    raw_fixture_frames: dict[str, pd.DataFrame],
    no_network: list[Any],
) -> None:
    cache = make_cache(tmp_path, fixture_dir, FIXTURE_TICKERS)
    panel = load_prices(offline_cfg(cache))

    for ticker in FIXTURE_TICKERS:
        expected = raw_fixture_frames[ticker]["close"]
        got = panel[ticker].reindex(expected.index)
        pd.testing.assert_series_equal(got, expected, check_names=False, check_freq=False)


# --------------------------------------------------------------------------- #
# 2. capped forward fill over SYNC's deliberate holes
# --------------------------------------------------------------------------- #
def test_sync_gaps_are_filled_only_up_to_the_ffill_limit(
    tmp_path: Path,
    fixture_dir: Path,
    raw_fixture_frames: dict[str, pd.DataFrame],
    no_network: list[Any],
) -> None:
    cache = make_cache(tmp_path, fixture_dir, FIXTURE_TICKERS)
    panel = load_prices(offline_cfg(cache, max_ffill_days=1))
    sync = panel["SYNC"]
    raw_close = raw_fixture_frames["SYNC"]["close"]

    # Exactly one day of each gap is carried forward; everything beyond stays NaN.
    still_missing = sorted(sync.index[sync.isna()])
    assert still_missing == [
        pd.Timestamp("2015-02-24"),
        pd.Timestamp("2016-07-15"),
        pd.Timestamp("2016-07-18"),
    ]

    for gap_start, length in SYNC_GAPS:
        first = pd.Timestamp(gap_start)
        prior = raw_close.loc[raw_close.index < first].iloc[-1]
        # The first day of every gap is filled with the last observed close...
        assert sync.loc[first] == pytest.approx(float(prior))
        # ...and a gap longer than the limit keeps its remaining days NaN.
        for offset in range(1, length):
            day = panel.index[panel.index.get_loc(first) + offset]
            assert np.isnan(sync.loc[day])

    # The fill never invents a price where one already exists.
    observed = sync.reindex(raw_close.index)
    pd.testing.assert_series_equal(observed, raw_close, check_names=False, check_freq=False)


def test_raising_the_ffill_limit_closes_every_sync_gap(
    tmp_path: Path, fixture_dir: Path, no_network: list[Any]
) -> None:
    cache = make_cache(tmp_path, fixture_dir, FIXTURE_TICKERS)
    tight = load_prices(offline_cfg(cache, max_ffill_days=1))
    loose = load_prices(offline_cfg(cache, max_ffill_days=3))

    assert tight["SYNC"].isna().sum() == 3
    assert loose["SYNC"].isna().sum() == 0
    assert loose.index.equals(tight.index)


def test_zero_ffill_limit_leaves_every_hole_open(
    tmp_path: Path, fixture_dir: Path, no_network: list[Any]
) -> None:
    cache = make_cache(tmp_path, fixture_dir, FIXTURE_TICKERS)
    panel = load_prices(offline_cfg(cache, max_ffill_days=0))
    assert sorted(panel.index[panel["SYNC"].isna()]) == SYNC_MISSING


# --------------------------------------------------------------------------- #
# 3. sparse-ticker removal (hand-built, no fixtures)
# --------------------------------------------------------------------------- #
def test_align_calendar_drops_sparse_ticker_and_keeps_borderline_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=100), name="date")
    sparse_dates = calendar.delete(list(range(10, 50)))  # 40 of 100 days missing
    borderline_dates = calendar.delete([50])  # 1 of 100 days missing

    frames = {
        "FULL": ohlcv(calendar, 100.0),
        "SPARSE": ohlcv(pd.DatetimeIndex(sparse_dates), 200.0),
        "BORDERLINE": ohlcv(pd.DatetimeIndex(borderline_dates), 300.0),
    }

    with caplog.at_level(logging.WARNING, logger="equity_forecast.data"):
        panel = align_calendar(frames, max_ffill_days=0, max_missing_frac=0.02)

    assert list(panel.columns) == ["FULL", "BORDERLINE"]
    assert panel.columns.name == "ticker"
    assert len(panel) == 100
    # BORDERLINE is kept complete with its single hole still open (no forward fill).
    assert panel["BORDERLINE"].isna().sum() == 1
    assert panel["FULL"].isna().sum() == 0

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "SPARSE" in warnings[0]
    assert "40.000%" in warnings[0]
    assert "BORDERLINE" not in warnings[0]


def test_borderline_ticker_is_rescued_by_the_forward_fill() -> None:
    """A ticker over the threshold on raw data survives if the fill can close its holes."""
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=100), name="date")
    holey = calendar.delete([20, 40, 60, 80, 90])  # 5% missing, all isolated single days

    frames = {"FULL": ohlcv(calendar), "HOLEY": ohlcv(pd.DatetimeIndex(holey), 50.0)}

    assert list(align_calendar(frames, 0, 0.02).columns) == ["FULL"]
    assert list(align_calendar(frames, 1, 0.02).columns) == ["FULL", "HOLEY"]


def test_align_calendar_raises_when_every_ticker_is_dropped() -> None:
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=100), name="date")
    frames = {
        "A": ohlcv(pd.DatetimeIndex(calendar.delete(list(range(0, 100, 2))))),
        "B": ohlcv(pd.DatetimeIndex(calendar.delete(list(range(1, 100, 2))))),
    }
    with pytest.raises(ValueError, match="every ticker was dropped"):
        align_calendar(frames, max_ffill_days=0, max_missing_frac=0.02)


def test_align_calendar_unions_calendars_and_trims_leading_nans() -> None:
    """A late-listing ticker that survives the threshold trims the panel's start."""
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=100), name="date")
    late = pd.DatetimeIndex(calendar[20:])
    extra_day = pd.DatetimeIndex([pd.Timestamp("2020-05-23")])  # a Saturday nobody else trades

    frames = {
        "OLD": ohlcv(calendar),
        "LATE": ohlcv(late, 500.0),
        "ODD": ohlcv(pd.DatetimeIndex(calendar.append(extra_day).sort_values()), 700.0),
    }
    panel = align_calendar(frames, max_ffill_days=0, max_missing_frac=0.3)

    assert list(panel.columns) == ["OLD", "LATE", "ODD"]
    # The union calendar picked up ODD's extra Saturday...
    assert extra_day[0] in panel.index
    # ...and the leading NaN block before LATE's first print is gone.
    assert panel.index[0] == calendar[20]
    assert not panel.iloc[0].isna().any()
    assert panel.notna().all(axis=1).iloc[0]


def test_align_calendar_rejects_empty_input_and_bad_arguments() -> None:
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=10), name="date")
    frames = {"A": ohlcv(calendar)}
    with pytest.raises(ValueError, match="at least one ticker frame"):
        align_calendar({}, 1, 0.02)
    with pytest.raises(ValueError, match="max_ffill_days"):
        align_calendar(frames, -1, 0.02)
    with pytest.raises(ValueError, match="max_missing_frac"):
        align_calendar(frames, 1, 1.5)
    with pytest.raises(KeyError, match="close"):
        align_calendar({"A": pd.DataFrame({"px": [1.0]}, index=calendar[:1])}, 1, 0.02)


# --------------------------------------------------------------------------- #
# 4. offline mode never touches the network
# --------------------------------------------------------------------------- #
def test_offline_missing_ticker_raises_without_calling_yfinance(
    tmp_path: Path, fixture_dir: Path, no_network: list[Any]
) -> None:
    cache = make_cache(tmp_path, fixture_dir, ("SYNA",))

    with pytest.raises(FileNotFoundError) as excinfo:
        load_prices(offline_cfg(cache, tickers=("SYNA", "SYNB", "MISSING")))

    message = str(excinfo.value)
    assert "SYNB" in message
    assert "MISSING" in message
    assert str(cache / "MISSING.parquet") in message
    assert "offline" in message
    assert no_network == []  # the tripwire was never tripped


def test_offline_refresh_is_rejected(tmp_path: Path, no_network: list[Any]) -> None:
    with pytest.raises(ValueError, match="offline"):
        download_prices(["SYNA"], "2015-01-01", None, tmp_path, refresh=True, offline=True)
    assert no_network == []


# --------------------------------------------------------------------------- #
# 5. the cache is authoritative
# --------------------------------------------------------------------------- #
def test_populated_cache_is_never_redownloaded(
    tmp_path: Path, fixture_dir: Path, no_network: list[Any]
) -> None:
    cache = make_cache(tmp_path, fixture_dir, FIXTURE_TICKERS)

    # offline=False, so the only thing stopping a download is the cache itself.
    frames = download_prices(FIXTURE_TICKERS, "2015-01-01", "2017-12-31", cache, offline=False)

    assert no_network == []
    assert list(frames) == list(FIXTURE_TICKERS)
    for ticker, frame in frames.items():
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert frame.index.name == "date"
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame["volume"].dtype == np.dtype("int64")
        assert frame["close"].dtype == np.dtype("float64")
        assert not frame.empty, ticker


@pytest.mark.parametrize("group_by", ["ticker", "column"])
def test_refresh_downloads_and_rewrites_the_cache(
    tmp_path: Path, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch, group_by: str
) -> None:
    cache = make_cache(tmp_path, fixture_dir, ("SYNA", "SYNB"))
    stale = pd.read_parquet(cache / "SYNA.parquet")
    calls: list[list[str]] = []

    def fake(tickers: list[str], **kwargs: Any) -> pd.DataFrame:
        calls.append(list(tickers))
        assert kwargs["auto_adjust"] is True
        return fake_yf_download(("SYNA", "SYNB"), group_by=group_by)

    monkeypatch.setattr(yfinance, "download", fake)
    frames = download_prices(("SYNA", "SYNB"), "2020-01-01", "2020-01-10", cache, refresh=True)

    assert calls == [["SYNA", "SYNB"]]
    assert list(frames["SYNA"]["close"]) == [1000.0 + i for i in range(6)]
    assert list(frames["SYNB"]["close"]) == [1100.0 + i for i in range(6)]
    assert frames["SYNA"]["volume"].dtype == np.dtype("int64")
    # The parquet on disk really was overwritten, not just the in-memory frame.
    assert not pd.read_parquet(cache / "SYNA.parquet").equals(stale)
    assert list(pd.read_parquet(cache / "SYNA.parquet").columns) == list(frames["SYNA"].columns)


def test_missing_ticker_is_downloaded_while_cached_ones_are_left_alone(
    tmp_path: Path, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = make_cache(tmp_path, fixture_dir, ("SYNA",))
    calls: list[list[str]] = []

    def fake(tickers: list[str], **kwargs: Any) -> pd.DataFrame:
        calls.append(list(tickers))
        return fake_yf_download(("NEW",), flat=True)

    monkeypatch.setattr(yfinance, "download", fake)
    frames = download_prices(("SYNA", "NEW"), "2020-01-01", None, cache, offline=False)

    assert calls == [["NEW"]]  # SYNA was cached, so only NEW went over the wire
    assert (cache / "NEW.parquet").exists()
    assert len(frames["SYNA"]) == 782
    assert list(frames["NEW"]["close"]) == [1000.0 + i for i in range(6)]


def test_empty_download_raises_runtime_error_naming_the_tickers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError) as excinfo:
        download_prices(("BOGUS", "ALSOBOGUS"), "2020-01-01", "2020-02-01", tmp_path / "c")
    assert "BOGUS" in str(excinfo.value)
    assert "ALSOBOGUS" in str(excinfo.value)


def test_download_returning_none_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="NADA"):
        download_prices(("NADA",), "2020-01-01", None, tmp_path / "c")


# --------------------------------------------------------------------------- #
# 6. date-range trimming and logging
# --------------------------------------------------------------------------- #
def test_load_prices_trims_to_the_requested_date_range(
    tmp_path: Path, fixture_dir: Path, no_network: list[Any]
) -> None:
    cache = make_cache(tmp_path, fixture_dir, FIXTURE_TICKERS)
    full = load_prices(offline_cfg(cache))
    window = load_prices(offline_cfg(cache, start="2016-03-01", end="2016-09-30"))

    assert window.index.min() >= pd.Timestamp("2016-03-01")
    assert window.index.max() <= pd.Timestamp("2016-09-30")
    assert len(window) < len(full)
    assert window.index.equals(
        full.index[
            (full.index >= pd.Timestamp("2016-03-01")) & (full.index <= pd.Timestamp("2016-09-30"))
        ]
    )
    # The end date is inclusive: 2016-09-30 was a Friday, so it must be present.
    assert pd.Timestamp("2016-09-30") in window.index
    pd.testing.assert_frame_equal(window, full.loc[window.index])


def test_open_ended_range_keeps_everything_from_start(
    tmp_path: Path, fixture_dir: Path, no_network: list[Any]
) -> None:
    cache = make_cache(tmp_path, fixture_dir, ("SYNA", "SYNB"))
    panel = load_prices(offline_cfg(cache, tickers=("SYNA", "SYNB"), start="2017-01-01", end=None))
    assert panel.index.min() >= pd.Timestamp("2017-01-01")
    assert panel.index.max() == pd.Timestamp("2017-12-29")


def test_every_load_is_logged_at_info_with_rows_and_date_range(
    tmp_path: Path,
    fixture_dir: Path,
    caplog: pytest.LogCaptureFixture,
    no_network: list[Any],
) -> None:
    cache = make_cache(tmp_path, fixture_dir, FIXTURE_TICKERS)
    with caplog.at_level(logging.INFO, logger="equity_forecast.data"):
        panel = load_prices(offline_cfg(cache))

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    for ticker, rows in (("SYNA", 782), ("SYNB", 782), ("SYNC", 776)):
        line = next(m for m in messages if m.startswith(f"loaded {ticker}:"))
        assert f"{rows} rows" in line
        assert "2015-01-01 -> 2017-12-29" in line
    assert any(f"{len(panel)} dates x 3 tickers" in m for m in messages)
