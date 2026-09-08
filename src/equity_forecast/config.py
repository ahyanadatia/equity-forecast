"""Typed configuration objects loaded from YAML.

Every knob the pipeline exposes lives in a YAML file (``configs/*.yaml``) and is
parsed into a frozen dataclass here. Nothing in the package reads raw
dictionaries, so a typo in a config file fails loudly at load time rather than
silently changing an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

__all__ = [
    "ArimaConfig",
    "ArimaGarchConfig",
    "BacktestConfig",
    "Config",
    "DataConfig",
    "FeatureConfig",
    "LstmConfig",
    "ModelsConfig",
    "ReportConfig",
    "RidgeConfig",
    "RunConfig",
    "SplitConfig",
    "load_config",
]

T = TypeVar("T")


@dataclass(frozen=True)
class RunConfig:
    """Global run settings."""

    seed: int = 42
    run_id: str | None = None


@dataclass(frozen=True)
class DataConfig:
    """Where prices come from and how they are cleaned."""

    tickers: tuple[str, ...] = ("AAPL", "JPM", "KO", "XOM", "MSFT", "PG")
    start: str = "2010-01-01"
    end: str | None = None
    cache_dir: str = "data/raw"
    offline: bool = False
    max_missing_frac: float = 0.02
    max_ffill_days: int = 1


@dataclass(frozen=True)
class FeatureConfig:
    """Target definition and feature construction."""

    horizon: int = 1
    n_lags: int = 10
    roll_windows: tuple[int, ...] = (5, 21, 63)
    drawdown_window: int = 21
    rvol_window: int = 21
    include_day_of_week: bool = True


@dataclass(frozen=True)
class SplitConfig:
    """Walk-forward (rolling-origin) split geometry."""

    mode: str = "expanding"
    initial_train_years: float = 5.0
    test_months: int = 6
    rolling_years: float = 5.0
    gap_days: int = 1

    def __post_init__(self) -> None:
        """Validate the split mode."""
        if self.mode not in {"expanding", "rolling"}:
            msg = f"splits.mode must be 'expanding' or 'rolling', got {self.mode!r}"
            raise ValueError(msg)
        if self.gap_days < 0:
            msg = f"splits.gap_days must be >= 0, got {self.gap_days}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ArimaConfig:
    """statsmodels ARIMA on returns (d = 0)."""

    order: tuple[int, int, int] = (1, 0, 1)
    auto: bool = False
    max_p: int = 2
    max_q: int = 2
    trend: str = "c"


@dataclass(frozen=True)
class ArimaGarchConfig:
    """arch ARX mean model with a GARCH(p, q) variance model."""

    mean_lags: int = 1
    dist: str = "t"
    p: int = 1
    q: int = 1


@dataclass(frozen=True)
class RidgeConfig:
    """Ridge regression on the lagged feature matrix."""

    alpha: float = 10.0
    fit_intercept: bool = True


@dataclass(frozen=True)
class LstmConfig:
    """Keras LSTM hyperparameters."""

    sequence_length: int = 20
    n_layers: int = 1
    units: int = 32
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 40
    batch_size: int = 128
    patience: int = 5
    val_frac: float = 0.15
    pooled: bool = True
    embedding_dim: int = 4


@dataclass(frozen=True)
class ModelsConfig:
    """Hyperparameters for every model, keyed by registry name."""

    zero_return: dict[str, Any] = field(default_factory=dict)
    historical_mean: dict[str, Any] = field(default_factory=dict)
    arima: ArimaConfig = field(default_factory=ArimaConfig)
    arima_garch: ArimaGarchConfig = field(default_factory=ArimaGarchConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    lstm: LstmConfig = field(default_factory=LstmConfig)


@dataclass(frozen=True)
class BacktestConfig:
    """Signal-to-position rules and cost assumptions."""

    cost_bps: float = 5.0
    threshold: float = 0.0
    vol_target: float | None = None
    vol_scale_cap: float = 2.0
    ann_factor: int = 252


@dataclass(frozen=True)
class ReportConfig:
    """Where run artefacts are written."""

    output_dir: str = "results"
    latest_dir: str = "results/latest"
    make_plots: bool = True


@dataclass(frozen=True)
class Config:
    """Top-level configuration for one experiment."""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


_TUPLE_FIELDS = {"tickers", "roll_windows", "order"}


def _build(cls: type[T], raw: object, path: str) -> T:
    """Instantiate a dataclass from a raw mapping, rejecting unknown keys.

    Args:
        cls: The dataclass to instantiate.
        raw: The mapping parsed from YAML, or ``None`` for "use all defaults". Typed as
            ``object`` because YAML can hand back anything at all.
        path: Dotted path of this section, used in error messages.

    Returns:
        An instance of ``cls`` with defaults filled in.

    Raises:
        TypeError: If ``raw`` is not a mapping.
        ValueError: If ``raw`` contains keys the dataclass does not define.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        msg = f"config section {path!r} must be a mapping, got {type(raw).__name__}"
        raise TypeError(msg)
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        msg = f"unknown key(s) in config section {path!r}: {sorted(unknown)}"
        raise ValueError(msg)

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        annotation = hints.get(name)
        if isinstance(annotation, type) and is_dataclass(annotation):
            kwargs[name] = _build(annotation, value, f"{path}.{name}")
        elif name in _TUPLE_FIELDS and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to a YAML config, e.g. ``configs/default.yaml``.

    Returns:
        A fully populated, frozen :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains keys the schema does not define.
    """
    p = Path(path)
    if not p.exists():
        msg = f"config file not found: {p}"
        raise FileNotFoundError(msg)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        msg = f"config file {p} must contain a YAML mapping at the top level"
        raise TypeError(msg)

    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        msg = f"unknown top-level config section(s): {sorted(unknown)}"
        raise ValueError(msg)

    return Config(
        run=_build(RunConfig, raw.get("run"), "run"),
        data=_build(DataConfig, raw.get("data"), "data"),
        features=_build(FeatureConfig, raw.get("features"), "features"),
        splits=_build(SplitConfig, raw.get("splits"), "splits"),
        models=_build(ModelsConfig, raw.get("models"), "models"),
        backtest=_build(BacktestConfig, raw.get("backtest"), "backtest"),
        report=_build(ReportConfig, raw.get("report"), "report"),
    )
