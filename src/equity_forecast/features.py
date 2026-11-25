"""Feature engineering: turn a wide price panel into the modelling dataset.

Everything the models see is produced here, and the single rule this module
exists to enforce is that **row ``t`` contains only information observable at
the close of ``t - 1``**. Features are therefore computed on the daily log
return series and then shifted forward by one day *within each ticker*, never
across tickers.

Notation: ``r_t = log(C_t) - log(C_{t-1})``, ``h = cfg.horizon``.

Dataset schema
--------------
:func:`build_dataset` returns a long frame with a ``(date, ticker)``
``pd.MultiIndex``, sorted, free of ``NaN``, all columns ``float64``, whose
columns are exactly ``[*feature_columns(cfg), "y", "ret_1d"]``:

===================  ========  ==================================================
column               kind      value at row ``t``
===================  ========  ==================================================
``ret_lag_k``        feature   ``r_{t-k}``, for ``k = 1 .. cfg.n_lags``
``roll_mean_w``      feature   ``mean(r_{t-w} .. r_{t-1})``, ``w`` in
                               ``cfg.roll_windows``
``roll_std_w``       feature   ``std(r_{t-w} .. r_{t-1}, ddof=1)``
``max_dd_w``         feature   max drawdown over ``r_{t-w} .. r_{t-1}``,
                               ``w = cfg.drawdown_window`` (positive, in [0, 1))
``rvol_w``           feature   ``sqrt(252 * mean(r^2 over t-w .. t-1))``,
                               ``w = cfg.rvol_window``
``dow_mon``..        feature   one-hot of ``t``'s weekday (known in advance);
``dow_fri``                    present only if ``cfg.include_day_of_week``
``y``                target    ``sum_{i=0}^{h-1} r_{t+i}`` (``h = 1`` -> ``ret_1d``)
``ret_1d``           audit     ``r_t``. Realised return on that date. **Never a
                               model input** - the backtest and evaluation use it
===================  ========  ==================================================

Column order is ``ret_lag_1 .. ret_lag_n``, then ``roll_mean_w, roll_std_w`` for
each ``w`` in ``cfg.roll_windows`` order, then ``max_dd_w``, ``rvol_w``, then the
five ``dow_*`` columns, then ``y`` and ``ret_1d``.

Max-drawdown definition
-----------------------
For the ``w`` returns ending at ``t - 1``, let ``c_j`` be the cumulative log
return after the ``j``-th of them and ``p_j = max(0, c_1, .., c_j)`` the running
peak of that path, which starts flat at 0 (i.e. the path is anchored to a
notional price of 1 immediately before the window). Then

    ``max_dd_w = 1 - exp(min_j (c_j - p_j))``

which is the largest peak-to-trough loss *as a fraction of the peak*, reported
as a positive number in ``[0, 1)``. It is 0 for a monotonically rising path.

Scaling
-------
Only the continuous columns are standardised; the ``dow_*`` one-hots are already
on a sane scale and are passed through untouched. :func:`fit_scaler` must be
given the **training rows of the current fold only** - the returned
:class:`ColumnScaler` is then applied to the whole fold window by
:func:`apply_scaler`, so no test-fold statistic ever leaks into training.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from equity_forecast.config import FeatureConfig

__all__ = [
    "DAY_OF_WEEK_COLUMNS",
    "FEATURE_SPLIT",
    "TARGET_COLUMNS",
    "TRADING_DAYS_PER_YEAR",
    "ColumnScaler",
    "apply_scaler",
    "binary_feature_columns",
    "build_dataset",
    "continuous_feature_columns",
    "feature_columns",
    "fit_scaler",
]

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR: Final[float] = 252.0
"""Annualisation factor for the realised-volatility feature."""

DAY_OF_WEEK_COLUMNS: Final[tuple[str, ...]] = (
    "dow_mon",
    "dow_tue",
    "dow_wed",
    "dow_thu",
    "dow_fri",
)
"""One-hot weekday columns, in Monday-to-Friday order."""

MARKET_LAGS: Final[int] = 5
"""Lags of the equal-weight market return added when ``market_features`` is on."""

TARGET_COLUMNS: Final[tuple[str, ...]] = ("y", "ret_1d")
"""Non-feature columns appended after the features, in order."""

FEATURE_SPLIT: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "continuous": (
            "ret_lag_",
            "roll_mean_",
            "roll_std_",
            "max_dd_",
            "rvol_",
            "mkt_",
            "rel_ret_",
            "abs_ret_",
            "vol_ratio",
        ),
        "binary": ("dow_",),
    }
)
"""Name prefixes splitting the features into scaled (continuous) and binary groups.

``FEATURE_SPLIT["binary"]`` is authoritative: a feature column is binary when it
starts with one of those prefixes and continuous otherwise, so the two groups are
always a partition of :func:`feature_columns`.
"""

_MIN_STD: Final[float] = 1e-12
"""Standard deviations at or below this are treated as zero variance."""


def _validate_config(cfg: FeatureConfig) -> None:
    """Reject feature settings that cannot produce a well-defined dataset.

    Args:
        cfg: The feature configuration to check.

    Raises:
        ValueError: If any window, lag count or horizon is out of range, or if
            ``roll_windows`` contains duplicates (which would collide as column
            names).
    """
    if cfg.horizon < 1:
        msg = f"features.horizon must be >= 1, got {cfg.horizon}"
        raise ValueError(msg)
    if cfg.n_lags < 1:
        msg = f"features.n_lags must be >= 1, got {cfg.n_lags}"
        raise ValueError(msg)
    if len(set(cfg.roll_windows)) != len(cfg.roll_windows):
        msg = f"features.roll_windows must not repeat a window, got {cfg.roll_windows}"
        raise ValueError(msg)
    for window in cfg.roll_windows:
        if window < 2:
            msg = f"every features.roll_windows entry must be >= 2, got {window}"
            raise ValueError(msg)
    if cfg.drawdown_window < 1:
        msg = f"features.drawdown_window must be >= 1, got {cfg.drawdown_window}"
        raise ValueError(msg)
    if cfg.rvol_window < 1:
        msg = f"features.rvol_window must be >= 1, got {cfg.rvol_window}"
        raise ValueError(msg)


def feature_columns(cfg: FeatureConfig) -> list[str]:
    """Return every model-input column implied by ``cfg``, in dataset order.

    Args:
        cfg: Feature configuration.

    Returns:
        The feature column names, ordered exactly as :func:`build_dataset`
        emits them.

    Raises:
        ValueError: If ``cfg`` is not a valid feature configuration.
    """
    _validate_config(cfg)
    columns = [f"ret_lag_{k}" for k in range(1, cfg.n_lags + 1)]
    for window in cfg.roll_windows:
        columns.append(f"roll_mean_{window}")
        columns.append(f"roll_std_{window}")
    columns.append(f"max_dd_{cfg.drawdown_window}")
    columns.append(f"rvol_{cfg.rvol_window}")
    if cfg.market_features:
        columns.extend(f"mkt_ret_lag_{k}" for k in range(1, MARKET_LAGS + 1))
        columns.append(f"mkt_rvol_{cfg.rvol_window}")
        columns.append("rel_ret_lag_1")
        columns.append("abs_ret_lag_1")
        if len(cfg.roll_windows) >= 2:
            columns.append("vol_ratio")
    if cfg.include_day_of_week:
        columns.extend(DAY_OF_WEEK_COLUMNS)
    return columns


def binary_feature_columns(cfg: FeatureConfig) -> list[str]:
    """Return the binary (never scaled) feature columns, in dataset order.

    Args:
        cfg: Feature configuration.

    Returns:
        The ``dow_*`` one-hot columns, or an empty list when
        ``cfg.include_day_of_week`` is ``False``.
    """
    prefixes = FEATURE_SPLIT["binary"]
    return [c for c in feature_columns(cfg) if c.startswith(prefixes)]


def continuous_feature_columns(cfg: FeatureConfig) -> list[str]:
    """Return the continuous (scaled) feature columns, in dataset order.

    Args:
        cfg: Feature configuration.

    Returns:
        Every feature column that is not binary; together with
        :func:`binary_feature_columns` this partitions :func:`feature_columns`.
    """
    prefixes = FEATURE_SPLIT["binary"]
    return [c for c in feature_columns(cfg) if not c.startswith(prefixes)]


def _validate_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Check the price panel and return it sorted by date.

    Args:
        prices: Wide panel, index ``date``, one column per ticker, adjusted
            close prices.

    Returns:
        A date-sorted copy of ``prices`` with ``float64`` values.

    Raises:
        TypeError: If the index is not a :class:`pandas.DatetimeIndex`.
        ValueError: If the panel has no columns, repeats a ticker, repeats a
            date, or contains a non-positive price (whose log is undefined).
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        msg = f"price panel must have a DatetimeIndex, got {type(prices.index).__name__}"
        raise TypeError(msg)
    if prices.shape[1] == 0:
        msg = "price panel has no ticker columns"
        raise ValueError(msg)
    if prices.columns.has_duplicates:
        msg = (
            f"price panel repeats ticker(s): {sorted(prices.columns[prices.columns.duplicated()])}"
        )
        raise ValueError(msg)
    if prices.index.has_duplicates:
        msg = "price panel index contains duplicate dates"
        raise ValueError(msg)
    panel = prices.sort_index().astype("float64")
    if bool((panel <= 0.0).to_numpy().any()):
        msg = "price panel contains non-positive prices; log returns are undefined"
        raise ValueError(msg)
    return panel


def _rolling_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    """Compute the max drawdown of the cumulative log-return path, rolling.

    The value at position ``t`` covers the ``window`` returns ending at ``t``
    inclusive; :func:`_ticker_block` shifts it by one day so that row ``t`` of the
    dataset sees only returns up to ``t - 1``. See the module docstring for the
    exact definition. Windows containing a ``NaN`` return yield ``NaN``.

    Args:
        returns: Daily log returns for one ticker.
        window: Number of returns in each window.

    Returns:
        A series aligned to ``returns`` holding a positive drawdown in [0, 1),
        ``NaN`` for the first ``window - 1`` positions.
    """
    values = returns.to_numpy(dtype=np.float64)
    n_obs = values.shape[0]
    out = np.full(n_obs, np.nan, dtype=np.float64)
    if n_obs >= window:
        blocks = sliding_window_view(values, window)
        cumulative = np.cumsum(blocks, axis=1)
        # The path starts at 0 (a notional price of 1), so the running peak is floored at 0
        # and a monotonically rising window scores exactly zero drawdown.
        peak = np.maximum(np.maximum.accumulate(cumulative, axis=1), 0.0)
        out[window - 1 :] = 1.0 - np.exp((cumulative - peak).min(axis=1))
    return pd.Series(out, index=returns.index)


def _day_of_week_columns(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Build the weekday one-hot columns for ``index``.

    The weekday of date ``t`` is known long before ``t``, so these columns need
    no shifting. Rows falling on a weekend (which a trading calendar should not
    contain) get all-zero one-hots rather than a spurious flag.

    Args:
        index: The dataset's date index.

    Returns:
        A mapping from column name to a 0/1 ``float64`` series.
    """
    weekday = np.asarray(index.weekday)
    return {
        name: pd.Series((weekday == position).astype(np.float64), index=index)
        for position, name in enumerate(DAY_OF_WEEK_COLUMNS)
    }


def _market_columns(
    returns: pd.Series, market: pd.Series, cfg: FeatureConfig
) -> dict[str, pd.Series]:
    """Cross-sectional context features for one ticker, all known at ``t - 1``.

    Args:
        returns: Daily log returns for the ticker.
        market: Equal-weight mean log return of the whole panel on each date.
        cfg: Feature configuration.

    Returns:
        The ``mkt_*``, ``rel_ret_lag_1``, ``abs_ret_lag_1`` and ``vol_ratio``
        columns, each shifted at least one day.
    """
    columns: dict[str, pd.Series] = {}
    for lag in range(1, MARKET_LAGS + 1):
        columns[f"mkt_ret_lag_{lag}"] = market.shift(lag)
    market_ms = market.pow(2).rolling(cfg.rvol_window).mean()
    columns[f"mkt_rvol_{cfg.rvol_window}"] = np.sqrt(TRADING_DAYS_PER_YEAR * market_ms).shift(1)
    columns["rel_ret_lag_1"] = (returns - market).shift(1)
    columns["abs_ret_lag_1"] = returns.abs().shift(1)
    if len(cfg.roll_windows) >= 2:
        short, long = min(cfg.roll_windows), max(cfg.roll_windows)
        short_std = returns.rolling(short).std(ddof=1)
        long_std = returns.rolling(long).std(ddof=1)
        ratio = short_std / long_std.where(long_std > _MIN_STD)
        columns["vol_ratio"] = ratio.shift(1)
    return columns


def _ticker_block(
    returns: pd.Series, cfg: FeatureConfig, market: pd.Series | None = None
) -> pd.DataFrame:
    """Build the feature/target block for a single ticker.

    Every rolling statistic is computed on ``returns`` and then shifted forward
    one day, so row ``t`` uses ``r_{t-1}`` and earlier and nothing else.

    Args:
        returns: Daily log returns for one ticker, indexed by date.
        cfg: Feature configuration.
        market: Equal-weight market return per date; required when
            ``cfg.market_features`` is on.

    Returns:
        A date-indexed frame with the feature columns followed by ``y`` and
        ``ret_1d``. Rows with insufficient history (and, for ``horizon > 1``, the
        final ``horizon - 1`` rows) contain ``NaN``.
    """
    columns: dict[str, pd.Series] = {}
    for lag in range(1, cfg.n_lags + 1):
        columns[f"ret_lag_{lag}"] = returns.shift(lag)
    for window in cfg.roll_windows:
        rolling = returns.rolling(window)
        columns[f"roll_mean_{window}"] = rolling.mean().shift(1)
        columns[f"roll_std_{window}"] = rolling.std(ddof=1).shift(1)
    columns[f"max_dd_{cfg.drawdown_window}"] = _rolling_max_drawdown(
        returns, cfg.drawdown_window
    ).shift(1)
    mean_square = returns.pow(2).rolling(cfg.rvol_window).mean()
    rvol: pd.Series = np.sqrt(TRADING_DAYS_PER_YEAR * mean_square)
    columns[f"rvol_{cfg.rvol_window}"] = rvol.shift(1)
    if cfg.market_features:
        if market is None:
            msg = "market_features needs the panel's market return"
            raise ValueError(msg)
        columns.update(_market_columns(returns, market, cfg))
    if cfg.include_day_of_week:
        index = returns.index
        if not isinstance(index, pd.DatetimeIndex):  # pragma: no cover - guarded upstream
            msg = "returns must be indexed by dates to build day-of-week features"
            raise TypeError(msg)
        columns.update(_day_of_week_columns(index))
    # y_t = sum_{i=0}^{h-1} r_{t+i}: a backward rolling sum pulled forward, which leaves the
    # last h-1 rows NaN because their horizon is not yet complete.
    columns["y"] = returns.rolling(cfg.horizon).sum().shift(-(cfg.horizon - 1))
    columns["ret_1d"] = returns
    return pd.DataFrame(columns, index=returns.index)


def build_dataset(prices: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Build the long modelling dataset from a wide price panel.

    Each ticker is processed independently - no per-ticker rolling statistic ever
    spans two tickers; the optional ``market_features`` are the one deliberate
    exception, built from the same-date cross-sectional mean return and shifted
    like everything else - and the row dated ``t`` is the row whose target is realised on
    ``t``, carrying features known at the close of ``t - 1``.

    Args:
        prices: Wide panel of adjusted closes, index ``date``, columns ``ticker``.
        cfg: Feature configuration.

    Returns:
        A frame with a ``(date, ticker)`` MultiIndex sorted by date then ticker,
        columns ``[*feature_columns(cfg), "y", "ret_1d"]``, dtype ``float64`` and
        no ``NaN``: rows without enough history, and the final ``horizon - 1``
        rows of each ticker, are dropped.

    Raises:
        TypeError: If ``prices`` is not indexed by dates.
        ValueError: If ``cfg`` or the panel is invalid.
    """
    _validate_config(cfg)
    panel = _validate_panel(prices)
    log_prices = pd.DataFrame(
        np.log(panel.to_numpy(dtype=np.float64)), index=panel.index, columns=panel.columns
    )
    returns = log_prices.diff()
    # Same-date mean across tickers; every feature built from it is shifted >= 1 day.
    market = returns.mean(axis=1, skipna=True)

    blocks: list[pd.DataFrame] = []
    for position, label in enumerate(panel.columns):
        ticker = str(label)
        block = _ticker_block(returns.iloc[:, position], cfg, market)
        block.index = pd.MultiIndex.from_product([block.index, [ticker]], names=["date", "ticker"])
        blocks.append(block)

    ordered = [*feature_columns(cfg), *TARGET_COLUMNS]
    dataset = pd.concat(blocks).loc[:, ordered].astype("float64")
    dataset = dataset.dropna(how="any").sort_index()
    logger.debug(
        "built dataset: %d rows, %d tickers, %d feature columns, horizon %d",
        len(dataset),
        panel.shape[1],
        len(ordered) - len(TARGET_COLUMNS),
        cfg.horizon,
    )
    return dataset


@dataclass(frozen=True)
class ColumnScaler:
    """Per-column mean and standard deviation of the continuous features.

    A deliberately tiny stand-in for scikit-learn's ``StandardScaler``: it holds
    plain tuples, so it is trivially typed, comparable and inspectable in tests,
    and it only ever touches the columns it was fitted on - binary ``dow_*``
    features and the ``y``/``ret_1d`` columns pass through :func:`apply_scaler`
    untouched.

    Attributes:
        columns: The scaled column names, in the order they were fitted.
        mean: Training mean of each column.
        std: Training standard deviation (``ddof=0``) of each column, with zero
            variance replaced by ``1.0`` so scaling can never divide by zero.
    """

    columns: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        """Check the three tuples line up and the deviations are usable.

        Raises:
            ValueError: If the tuples differ in length, or if any standard
                deviation is not finite and strictly positive.
        """
        if not len(self.columns) == len(self.mean) == len(self.std):
            msg = (
                "ColumnScaler needs one mean and one std per column, got "
                f"{len(self.columns)} columns, {len(self.mean)} means, {len(self.std)} stds"
            )
            raise ValueError(msg)
        for name, value in zip(self.columns, self.std, strict=True):
            if not np.isfinite(value) or value <= 0.0:
                msg = f"ColumnScaler std for {name!r} must be finite and positive, got {value!r}"
                raise ValueError(msg)

    def to_frame(self) -> pd.DataFrame:
        """Return the fitted statistics as a frame indexed by column name.

        Returns:
            A frame with index ``column`` and columns ``mean`` and ``std``.
        """
        return pd.DataFrame(
            {"mean": list(self.mean), "std": list(self.std)},
            index=pd.Index(list(self.columns), name="column"),
        )


def _require_columns(frame: pd.DataFrame, columns: list[str], what: str) -> None:
    """Raise if ``frame`` is missing any of ``columns``.

    Args:
        frame: The frame to check.
        columns: Required column names.
        what: Short description of the frame, used in the error message.

    Raises:
        KeyError: If any required column is absent.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        msg = f"{what} is missing column(s): {missing}"
        raise KeyError(msg)


def fit_scaler(x_train: pd.DataFrame, cfg: FeatureConfig) -> ColumnScaler:
    """Fit per-column standardisation statistics on training rows only.

    Args:
        x_train: Feature frame holding **only** the current fold's training
            rows. Passing anything else leaks future information into training.
        cfg: Feature configuration, which names the continuous columns.

    Returns:
        A :class:`ColumnScaler` over :func:`continuous_feature_columns`.

    Raises:
        KeyError: If ``x_train`` lacks a continuous feature column.
        ValueError: If ``x_train`` is empty or holds a non-finite value.
    """
    columns = continuous_feature_columns(cfg)
    _require_columns(x_train, columns, "training feature frame")
    if len(x_train) == 0:
        msg = "cannot fit a scaler on an empty training frame"
        raise ValueError(msg)

    block = x_train.loc[:, columns].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(block).all()):
        msg = "training feature frame contains NaN or infinite values"
        raise ValueError(msg)

    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=0)
    degenerate = ~np.isfinite(std) | (std <= _MIN_STD)
    if bool(degenerate.any()):
        flat = [c for c, bad in zip(columns, degenerate.tolist(), strict=True) if bad]
        logger.warning("zero-variance feature(s) left unscaled: %s", flat)
    guarded = np.where(degenerate, 1.0, std)
    return ColumnScaler(
        columns=tuple(columns),
        mean=tuple(float(v) for v in mean),
        std=tuple(float(v) for v in guarded),
    )


def apply_scaler(scaler: ColumnScaler, x: pd.DataFrame) -> pd.DataFrame:
    """Standardise the scaler's columns, leaving every other column alone.

    The index, the column order and the values of untouched columns (binary
    ``dow_*`` features, ``y``, ``ret_1d``, anything else) are preserved exactly.

    Args:
        scaler: Statistics fitted by :func:`fit_scaler` on training rows.
        x: Any frame containing the scaler's columns - typically the whole
            fold window (train + gap + test rows).

    Returns:
        A new frame, scaled in place of the continuous columns.

    Raises:
        KeyError: If ``x`` lacks one of the scaler's columns.
    """
    columns = list(scaler.columns)
    _require_columns(x, columns, "feature frame")
    out = x.copy()
    if not columns:
        return out
    block = out.loc[:, columns].to_numpy(dtype=np.float64)
    scaled = (block - np.asarray(scaler.mean, dtype=np.float64)) / np.asarray(
        scaler.std, dtype=np.float64
    )
    out.loc[:, columns] = scaled
    return out
