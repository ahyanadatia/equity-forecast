"""Generate the simulated equity universes used for the committed reference runs.

Why this exists
---------------
The headline study is meant to run on real prices from Yahoo Finance. The
environment this repository was built in blocks market-data egress, so the
committed reference results come from simulation instead. Rather than shrink the
study, this script builds two universes at the full scale of
``configs/default.yaml`` - six tickers, 2010 to 2026 - so every part of the
pipeline is exercised over 20-odd walk-forward folds.

The two universes answer two different questions:

``efficient``
    A near-efficient market: a common factor plus idiosyncratic noise, both with
    GARCH(1,1) conditional variance and Student-t innovations, and **no
    exploitable predictability** in the conditional mean. If a model appears to
    beat the random walk here, the harness is broken.

``signal``
    Identical, except an AR(1) term (``phi = 0.15``) is planted in each stock's
    return process. That is far stronger than anything daily equity returns
    exhibit, and deliberately so: the point of a positive control is to be
    unambiguously detectable, not to be realistic. Something genuinely
    forecastable is present, so a working harness must find it. This is what
    makes a null result on the efficient universe evidence about the data rather
    than evidence about the code.

Tickers are named ``SIMA``-``SIMF`` so no output can be mistaken for a real
security. Output is byte-stable for a given seed.

Usage::

    python scripts/make_synthetic_universe.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

START = "2010-01-01"
END = "2026-09-01"

TICKERS: tuple[str, ...] = ("SIMA", "SIMB", "SIMC", "SIMD", "SIME", "SIMF")
# (ticker, market beta, annualised idiosyncratic drift, idiosyncratic vol level, start price)
SPECS: tuple[tuple[str, float, float, float, float], ...] = (
    ("SIMA", 1.15, 0.04, 0.9, 30.0),
    ("SIMB", 1.30, 0.01, 1.2, 45.0),
    ("SIMC", 0.65, 0.02, 0.7, 60.0),
    ("SIMD", 0.90, -0.01, 1.0, 75.0),
    ("SIME", 1.05, 0.03, 0.8, 25.0),
    ("SIMF", 0.55, 0.02, 0.6, 90.0),
)

MARKET_SEED = 20100104
MARKET_DRIFT = 0.06
MARKET_GARCH = (2.0e-6, 0.09, 0.89)  # omega, alpha, beta
IDIO_GARCH = (3.0e-6, 0.07, 0.90)
NU = 5.0
SIGNAL_PHI = 0.15


def _standardised_t(rng: np.random.Generator, n: int, nu: float = NU) -> np.ndarray:
    """Draw ``n`` Student-t innovations rescaled to unit variance."""
    return rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2.0))


def _garch_path(
    rng: np.random.Generator, n: int, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """Simulate a mean-zero GARCH(1, 1) innovation path with Student-t shocks."""
    z = _standardised_t(rng, n)
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-6)
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    return eps


def simulate_universe(
    dates: pd.DatetimeIndex, *, phi: float, seed_offset: int
) -> dict[str, pd.DataFrame]:
    """Simulate one universe of correlated stocks.

    Args:
        dates: Trading calendar.
        phi: AR(1) coefficient planted in each stock's conditional mean. ``0.0``
            gives a near-efficient market; a small positive value plants a real,
            findable signal.
        seed_offset: Added to every seed so the two universes are independent.

    Returns:
        Mapping of ticker to an OHLCV frame in the canonical cache schema.
    """
    n = len(dates)
    market_rng = np.random.default_rng(MARKET_SEED + seed_offset)
    market = MARKET_DRIFT / 252.0 + _garch_path(market_rng, n, *MARKET_GARCH)

    frames: dict[str, pd.DataFrame] = {}
    for i, (ticker, beta, drift, idio_scale, px0) in enumerate(SPECS):
        rng = np.random.default_rng(MARKET_SEED + seed_offset + 977 * (i + 1))
        idio = idio_scale * _garch_path(rng, n, *IDIO_GARCH)
        mu = drift / 252.0

        # The AR(1) term acts on the stock's own past return, so a model with
        # lagged returns in its feature set has something real to find.
        ret = np.empty(n)
        ret[0] = mu + beta * market[0] + idio[0]
        for t in range(1, n):
            ret[t] = mu + phi * (ret[t - 1] - mu) + beta * market[t] + idio[t]

        close = px0 * np.exp(np.cumsum(ret))
        intraday = np.abs(ret) + 0.004
        open_ = close * np.exp(-ret + rng.normal(0.0, 0.25 * intraday, n))
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.5 * intraday, n)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.5 * intraday, n)))
        volume = np.round(np.exp(rng.normal(15.5, 0.4, n))).astype("int64")

        frame = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        frame.index.name = "date"
        frames[ticker] = frame
    return frames


def main() -> None:
    """Write both universes to their cache directories."""
    calendar = pd.bdate_range(START, END, name="date")
    for name, phi, offset in (("synthetic", 0.0, 0), ("synthetic_signal", SIGNAL_PHI, 5_000_000)):
        out_dir = ROOT / "data" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = simulate_universe(calendar, phi=phi, seed_offset=offset)
        for ticker, frame in frames.items():
            frame.to_parquet(out_dir / f"{ticker}.parquet", engine="pyarrow", compression="snappy")
        closes = pd.DataFrame({t: f["close"] for t, f in frames.items()})
        rets = np.log(closes).diff().dropna()
        ann_vol = float(rets.to_numpy().std(ddof=1) * np.sqrt(252))
        mean_corr = float(rets.corr().to_numpy()[np.triu_indices(len(TICKERS), 1)].mean())
        ar1 = float(np.mean([rets[t].autocorr(1) for t in TICKERS]))
        print(
            f"{name}: {len(calendar)} days x {len(frames)} tickers -> {out_dir} "
            f"| ann vol {ann_vol:.1%} | mean pairwise corr {mean_corr:.2f} | mean AR(1) {ar1:+.4f}"
        )


if __name__ == "__main__":
    main()
