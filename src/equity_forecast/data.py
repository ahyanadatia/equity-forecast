"""Price acquisition, on-disk caching and trading-calendar alignment.

Three responsibilities, deliberately kept separable:

``download_prices``
    Materialise one parquet per ticker under ``cache_dir`` in the canonical cache
    schema (``date`` index; ``open, high, low, close, volume`` columns). The cache
    is authoritative: a ticker whose parquet already exists is *never* re-fetched
    unless ``refresh=True``, and with ``offline=True`` the network is not touched
    at all.

``align_calendar``
    The pure core. Given per-ticker frames it builds the wide close-price panel on
    the union trading calendar, forward-fills short holes, and drops tickers whose
    history is too full of holes to be usable. It takes no config object and does
    no I/O, so the awkward cases (ragged calendars, long gaps, late listings) are
    testable without a network or even a filesystem.

``load_prices``
    The thin glue the pipeline actually calls: cache -> trim to the requested date
    range -> ``align_calendar``.

Why forward-fill is capped
--------------------------
A one-day hole in a vendor feed is a data glitch, and carrying the previous close
across it is harmless. A two-week hole is a halted stock, and carrying the last
close across it fabricates a fortnight of exactly-zero returns - which looks like
a wonderfully low-volatility asset to any model downstream. So the fill is capped
at ``max_ffill_days`` consecutive days and anything longer is left as ``NaN``,
where it will be dropped by the feature builder rather than silently believed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from equity_forecast.config import DataConfig

__all__ = [
    "OHLCV_COLUMNS",
    "align_calendar",
    "download_prices",
    "load_prices",
]

logger = logging.getLogger(__name__)

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
"""The canonical cache columns, in order."""

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    """Return the parquet path for one ticker.

    Args:
        cache_dir: Directory holding the price cache.
        ticker: Ticker symbol, already normalised.

    Returns:
        ``<cache_dir>/<TICKER>.parquet``.
    """
    return cache_dir / f"{ticker}.parquet"


def _normalise_tickers(tickers: Sequence[str]) -> list[str]:
    """Upper-case, strip and de-duplicate a ticker list, preserving order.

    Args:
        tickers: Raw ticker symbols.

    Returns:
        The cleaned symbols in their original order.

    Raises:
        ValueError: If ``tickers`` is empty or contains only blanks.
    """
    seen: dict[str, None] = {}
    for raw in tickers:
        symbol = str(raw).strip().upper()
        if symbol:
            seen[symbol] = None
    if not seen:
        msg = "no tickers requested: `tickers` must contain at least one non-blank symbol"
        raise ValueError(msg)
    return list(seen)


def _normalise_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Coerce a raw OHLCV frame into the canonical cache schema.

    Column names are lower-cased (yfinance yields ``Open``/``Close``/...), the
    index is turned into a tz-naive ``DatetimeIndex`` named ``date`` normalised to
    midnight, duplicates are collapsed and dtypes are pinned to float64 (int64 for
    volume).

    Args:
        frame: Raw per-ticker frame from yfinance or from the parquet cache.
        ticker: Ticker the frame belongs to, used in error messages.

    Returns:
        A frame with exactly the columns in :data:`OHLCV_COLUMNS`, sorted by date.

    Raises:
        RuntimeError: If any canonical OHLCV column is absent.
    """
    out = frame.copy()
    out.columns = pd.Index([str(name).strip().lower() for name in out.columns])
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    absent = [column for column in OHLCV_COLUMNS if column not in out.columns]
    if absent:
        msg = (
            f"price data for {ticker!r} is missing column(s) {absent}; "
            f"got {sorted(out.columns)}. Expected the OHLCV schema {list(OHLCV_COLUMNS)}."
        )
        raise RuntimeError(msg)
    out = out.loc[:, list(OHLCV_COLUMNS)]

    index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if index.tz is not None:
        # Keep the exchange-local calendar date rather than shifting it into UTC.
        index = index.tz_localize(None)
    out.index = index.normalize()
    out.index.name = "date"

    out = out[~out.index.duplicated(keep="last")].sort_index()
    for column in _PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    out = out.dropna(subset=["close"])
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out["volume"] = volume.round().astype("int64")
    return out


def _split_download(raw: pd.DataFrame, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Split a yfinance download into one frame per ticker.

    yfinance returns a flat column index for a single-ticker request and a column
    ``MultiIndex`` for a multi-ticker one - with the ticker on either level
    depending on ``group_by``. Both shapes are handled here.

    Args:
        raw: The frame ``yfinance.download`` returned.
        tickers: The tickers that were requested.

    Returns:
        A mapping of ticker to its own (still un-normalised) OHLCV frame.

    Raises:
        RuntimeError: If a flat frame comes back for a multi-ticker request.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        if len(tickers) != 1:
            msg = (
                f"yfinance returned a flat column index for {len(tickers)} tickers "
                f"{list(tickers)}; cannot tell the series apart"
            )
            raise RuntimeError(msg)
        return {tickers[0]: raw}

    wanted = {ticker.upper() for ticker in tickers}
    level = 0 if wanted & {str(v).upper() for v in raw.columns.get_level_values(0)} else 1
    field_level = 1 - level

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        columns = [c for c in raw.columns if str(c[level]).upper() == ticker.upper()]
        if not columns:
            continue
        block = raw.loc[:, columns].copy()
        block.columns = pd.Index([str(c[field_level]) for c in columns])
        out[ticker] = block
    return out


def _fetch(tickers: Sequence[str], start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
    """Download OHLCV history from Yahoo Finance.

    Args:
        tickers: Symbols to fetch.
        start: Inclusive ISO start date, or ``None`` for "as far back as available".
        end: Inclusive ISO end date, or ``None`` for "up to today". yfinance treats
            ``end`` as exclusive, so one day is added before the call.

    Returns:
        A mapping of ticker to raw frame.

    Raises:
        RuntimeError: If yfinance returns nothing usable for the whole request.
    """
    import yfinance as yf  # noqa: PLC0415  (deferred: keeps offline runs import-free)

    exclusive_end = None if end is None else (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    logger.info("downloading %d ticker(s) from yfinance: %s", len(tickers), list(tickers))
    raw = yf.download(
        list(tickers),
        start=start,
        end=exclusive_end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if raw is None or len(raw.index) == 0 or raw.empty:
        msg = (
            f"yfinance returned no data for {list(tickers)} over "
            f"[{start or 'inception'}, {end or 'today'}]. Check the symbols and the date range, "
            f"or re-run with offline=True against an existing cache."
        )
        raise RuntimeError(msg)
    return _split_download(raw, tickers)


def download_prices(  # noqa: PLR0917 - the signature is fixed by the module contract
    tickers: Sequence[str],
    start: str | None,
    end: str | None,
    cache_dir: str | Path,
    refresh: bool = False,
    offline: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return daily adjusted OHLCV per ticker, fetching only what is not cached.

    Each ticker is stored at ``<cache_dir>/<TICKER>.parquet`` in the canonical
    cache schema. A ticker whose parquet exists is read straight from disk and is
    never re-downloaded unless ``refresh`` is set, so repeated runs are both fast
    and reproducible.

    Args:
        tickers: Symbols to load. Case-insensitive; normalised to upper case.
        start: Inclusive ISO start date for downloads, or ``None``.
        end: Inclusive ISO end date for downloads, or ``None`` for today.
        cache_dir: Directory holding (or to hold) the parquet cache.
        refresh: If ``True``, re-download every ticker and overwrite the cache.
        offline: If ``True``, never touch the network. Every requested ticker must
            already be cached.

    Returns:
        A mapping of upper-cased ticker to its OHLCV frame, in request order.

    Raises:
        FileNotFoundError: If ``offline`` is set and a ticker has no cache file.
        ValueError: If ``offline`` and ``refresh`` are both set, or no ticker is given.
        RuntimeError: If a download comes back empty for some requested tickers.
    """
    symbols = _normalise_tickers(tickers)
    cache = Path(cache_dir)

    if offline and refresh:
        msg = "cannot refresh prices with offline=True: refreshing requires the network"
        raise ValueError(msg)

    uncached = [t for t in symbols if not _cache_path(cache, t).exists()]
    if offline:
        if uncached:
            paths = ", ".join(str(_cache_path(cache, t)) for t in uncached)
            msg = (
                f"offline=True but {len(uncached)} ticker(s) are not cached: {uncached}. "
                f"Expected parquet file(s) at: {paths}. "
                f"Re-run once with data.offline=false (or call download_prices with "
                f"offline=False) to populate the cache, or point data.cache_dir at a "
                f"directory that already has them."
            )
            raise FileNotFoundError(msg)
    else:
        to_fetch = symbols if refresh else uncached
        if to_fetch:
            cache.mkdir(parents=True, exist_ok=True)
            fetched = _fetch(to_fetch, start, end)
            empty: list[str] = []
            for ticker in to_fetch:
                block = fetched.get(ticker)
                frame = _normalise_frame(block, ticker) if block is not None else None
                if frame is None or frame.empty:
                    empty.append(ticker)
                    continue
                frame.to_parquet(_cache_path(cache, ticker), engine="pyarrow")
            if empty:
                msg = (
                    f"yfinance returned no usable rows for {empty} over "
                    f"[{start or 'inception'}, {end or 'today'}]. "
                    f"Check the symbols and the date range."
                )
                raise RuntimeError(msg)

    out: dict[str, pd.DataFrame] = {}
    for ticker in symbols:
        path = _cache_path(cache, ticker)
        frame = _normalise_frame(pd.read_parquet(path), ticker)
        out[ticker] = frame
        _log_frame(ticker, frame, path)
    return out


def _log_frame(ticker: str, frame: pd.DataFrame, path: Path) -> None:
    """Emit the FR-D4 INFO line for one loaded ticker.

    Args:
        ticker: Ticker symbol.
        frame: The loaded OHLCV frame.
        path: Cache file the frame came from.
    """
    if frame.empty:
        logger.info("loaded %s: 0 rows from %s", ticker, path)
        return
    logger.info(
        "loaded %s: %d rows %s -> %s from %s",
        ticker,
        len(frame),
        frame.index[0].date(),
        frame.index[-1].date(),
        path,
    )


def _close_series(frame: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract the adjusted close series from one ticker's frame.

    Args:
        frame: OHLCV frame (or any frame carrying a ``close`` column).
        ticker: Ticker the frame belongs to, used in error messages.

    Returns:
        A float64 series indexed by a sorted, de-duplicated ``DatetimeIndex``.

    Raises:
        KeyError: If the frame has no ``close`` column.
    """
    lowered = {str(name).lower(): name for name in frame.columns}
    if "close" not in lowered:
        msg = f"price frame for {ticker!r} has no 'close' column; got {sorted(frame.columns)}"
        raise KeyError(msg)
    values = pd.to_numeric(frame[lowered["close"]], errors="coerce").to_numpy(dtype="float64")

    index = pd.DatetimeIndex(pd.to_datetime(pd.Index(frame.index)))
    if index.tz is not None:
        index = index.tz_localize(None)

    series: pd.Series = pd.Series(values, index=index.normalize(), name=ticker, dtype="float64")
    unique: pd.Series = series[~series.index.duplicated(keep="last")]
    return unique.sort_index()


def align_calendar(
    frames: Mapping[str, pd.DataFrame],
    max_ffill_days: int,
    max_missing_frac: float,
) -> pd.DataFrame:
    """Build the wide close-price panel on the union trading calendar.

    The steps, in order:

    1. Union every ticker's dates into one calendar and reindex each close series
       onto it, so ragged vendor calendars line up.
    2. Forward-fill at most ``max_ffill_days`` *consecutive* days per ticker.
       Leading gaps - dates before a ticker's history starts - are never filled,
       because there is nothing to carry forward.
    3. Drop any ticker still missing more than ``max_missing_frac`` of the
       calendar, logging a warning that names it and its missing fraction.
    4. Drop dates on which no surviving ticker traded (they can only have come
       from a dropped ticker's calendar) and trim the leading edge to the first
       date on which every surviving ticker has a price, so the panel starts
       fully populated.

    Args:
        frames: Mapping of ticker to a frame carrying a ``close`` column.
        max_ffill_days: Maximum run of consecutive days to carry a price across.
            ``0`` disables forward filling entirely.
        max_missing_frac: Fraction of the calendar a ticker may still be missing
            after the fill before it is dropped, in ``[0, 1]``.

    Returns:
        A wide float64 frame: index a ``DatetimeIndex`` named ``date``, columns a
        ``pd.Index`` of tickers named ``ticker``, values the adjusted close.

    Raises:
        ValueError: If ``frames`` is empty, the arguments are out of range, no
            ticker carries any dates, or every ticker is dropped as too sparse.
    """
    if not frames:
        msg = "align_calendar needs at least one ticker frame, got none"
        raise ValueError(msg)
    if max_ffill_days < 0:
        msg = f"max_ffill_days must be >= 0, got {max_ffill_days}"
        raise ValueError(msg)
    if not 0.0 <= max_missing_frac <= 1.0:
        msg = f"max_missing_frac must lie in [0, 1], got {max_missing_frac}"
        raise ValueError(msg)

    closes = {ticker: _close_series(frame, ticker) for ticker, frame in frames.items()}

    stamps: set[pd.Timestamp] = set()
    for series in closes.values():
        stamps.update(pd.Timestamp(ts) for ts in series.index)
    calendar = pd.DatetimeIndex(sorted(stamps), name="date")
    if calendar.empty:
        msg = f"no dates in any of the supplied price frames: {sorted(frames)}"
        raise ValueError(msg)

    panel = pd.DataFrame(
        {ticker: series.reindex(calendar) for ticker, series in closes.items()},
        index=calendar,
        dtype="float64",
    )
    panel.columns = pd.Index(list(closes), name="ticker")

    filled = panel.ffill(limit=max_ffill_days) if max_ffill_days > 0 else panel

    n_days = len(calendar)
    fracs = (filled.isna().sum(axis=0) / n_days).to_numpy(dtype="float64")
    keep: list[str] = []
    for ticker, raw_frac in zip(list(closes), fracs, strict=True):
        frac = float(raw_frac)
        if frac > max_missing_frac:
            logger.warning(
                "dropping %s: %.3f%% of the %d-day calendar is still missing after a "
                "%d-day forward fill (limit %.3f%%)",
                ticker,
                100.0 * frac,
                n_days,
                max_ffill_days,
                100.0 * max_missing_frac,
            )
        else:
            keep.append(str(ticker))

    if not keep:
        msg = (
            f"every ticker was dropped: none of {sorted(frames)} covers more than "
            f"{100.0 * (1.0 - max_missing_frac):.3f}% of the {n_days}-day calendar after a "
            f"{max_ffill_days}-day forward fill. Widen data.max_missing_frac / "
            f"data.max_ffill_days, or narrow the requested date range."
        )
        raise ValueError(msg)

    out = filled.loc[:, keep].dropna(how="all")
    fully_populated = out.notna().all(axis=1).to_numpy(dtype="bool")
    if not fully_populated.any():  # pragma: no cover - defensive
        msg = f"the surviving tickers {keep} have no date on which all of them have a price"
        raise ValueError(msg)
    out = out.iloc[int(fully_populated.argmax()) :]

    out.index = pd.DatetimeIndex(out.index, name="date")
    out.columns = pd.Index(keep, name="ticker")
    return out.astype("float64")


def load_prices(cfg: DataConfig, refresh: bool = False) -> pd.DataFrame:
    """Load the wide close-price panel described by a :class:`DataConfig`.

    Reads (and, unless ``cfg.offline``, populates) the parquet cache, trims each
    ticker to ``[cfg.start, cfg.end]``, then hands the frames to
    :func:`align_calendar` for calendar alignment, capped forward filling and
    sparse-ticker removal.

    Args:
        cfg: The ``data:`` section of the experiment config.
        refresh: If ``True``, re-download every ticker even when it is cached.

    Returns:
        A wide float64 panel: index a ``DatetimeIndex`` named ``date``, columns a
        ``pd.Index`` of tickers named ``ticker``, values the adjusted close.

    Raises:
        FileNotFoundError: If ``cfg.offline`` is set and a ticker is not cached.
        ValueError: If every requested ticker is dropped as too sparse.
    """
    frames = download_prices(
        cfg.tickers,
        cfg.start,
        cfg.end,
        cfg.cache_dir,
        refresh=refresh,
        offline=cfg.offline,
    )

    start = pd.Timestamp(cfg.start).normalize() if cfg.start else None
    end = pd.Timestamp(cfg.end).normalize() if cfg.end else None
    trimmed = {ticker: _trim(frame, start, end) for ticker, frame in frames.items()}

    panel = align_calendar(trimmed, cfg.max_ffill_days, cfg.max_missing_frac)
    dropped = [t for t in frames if t not in panel.columns]
    logger.info(
        "price panel: %d dates x %d tickers %s -> %s (%d dropped: %s)",
        len(panel.index),
        len(panel.columns),
        panel.index[0].date(),
        panel.index[-1].date(),
        len(dropped),
        dropped or "none",
    )
    return panel


def _trim(
    frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> pd.DataFrame:
    """Restrict a date-indexed frame to the inclusive window ``[start, end]``.

    Args:
        frame: A frame with a ``DatetimeIndex``.
        start: Inclusive lower bound, or ``None`` for no lower bound.
        end: Inclusive upper bound, or ``None`` for no upper bound.

    Returns:
        The rows falling inside the window.
    """
    out = frame
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out
