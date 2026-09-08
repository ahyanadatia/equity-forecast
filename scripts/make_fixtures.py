"""Generate the deterministic synthetic price fixtures used by the test suite.

The test suite must run without network access (NFR-4), so it cannot depend on
Yahoo Finance. Instead it runs against synthetic tickers produced by this
script from a data-generating process that shares the awkward features of real
daily equity returns:

* GARCH(1,1) conditional variance, so volatility clusters;
* Student-t(5) standardised innovations, so returns are fat-tailed;
* a very small AR(1) term, so there is a whisper of predictability but nothing
  a model can comfortably exploit;
* a small positive drift.

Re-run with ``python scripts/make_fixtures.py`` - the output is byte-stable for
a given seed, so regenerating never produces a spurious diff.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "prices"

# (ticker, seed, annualised drift, omega, alpha, beta, ar1, start price)
SPECS: tuple[tuple[str, int, float, float, float, float, float, float], ...] = (
    ("SYNA", 20150101, 0.08, 2.0e-6, 0.08, 0.90, 0.02, 100.0),
    ("SYNB", 20150102, 0.04, 3.0e-6, 0.10, 0.87, -0.03, 45.0),
    ("SYNC", 20150103, 0.06, 2.5e-6, 0.06, 0.91, 0.00, 210.0),
)

START = "2015-01-01"
END = "2017-12-31"


def simulate_prices(
    dates: pd.DatetimeIndex,
    *,
    seed: int,
    ann_drift: float,
    omega: float,
    alpha: float,
    beta: float,
    ar1: float,
    start_price: float,
) -> pd.DataFrame:
    """Simulate one ticker's daily OHLCV history.

    Args:
        dates: Trading calendar to simulate over.
        seed: Seed for this ticker's generator.
        ann_drift: Annualised drift of the log-return process.
        omega: GARCH constant.
        alpha: GARCH ARCH coefficient.
        beta: GARCH lag coefficient.
        ar1: AR(1) coefficient of the conditional mean.
        start_price: Price on the first date.

    Returns:
        A frame indexed by date with ``open``, ``high``, ``low``, ``close`` and
        ``volume`` columns.
    """
    rng = np.random.default_rng(seed)
    n = len(dates)
    nu = 5.0
    # Standardised Student-t innovations (unit variance).
    z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2.0))

    mu = ann_drift / 252.0
    sigma2 = np.empty(n)
    eps = np.empty(n)
    ret = np.empty(n)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-6)
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    ret[0] = mu + eps[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
        ret[t] = mu + ar1 * (ret[t - 1] - mu) + eps[t]

    close = start_price * np.exp(np.cumsum(ret))
    intraday = np.sqrt(sigma2)
    open_ = close * np.exp(-ret + rng.normal(0.0, 0.2 * intraday, n))
    hi_noise = np.abs(rng.normal(0.0, 0.6 * intraday, n))
    lo_noise = np.abs(rng.normal(0.0, 0.6 * intraday, n))
    high = np.maximum(open_, close) * np.exp(hi_noise)
    low = np.minimum(open_, close) * np.exp(-lo_noise)
    volume = np.round(np.exp(rng.normal(15.0, 0.35, n))).astype("int64")

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    frame.index.name = "date"
    floats = {"open": "float64", "high": "float64", "low": "float64", "close": "float64"}
    return frame.astype(floats)


def main() -> None:
    """Write every fixture parquet to ``tests/fixtures/prices``."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    calendar = pd.bdate_range(START, END, name="date")
    for ticker, seed, drift, omega, alpha, beta, ar1, px0 in SPECS:
        frame = simulate_prices(
            calendar,
            seed=seed,
            ann_drift=drift,
            omega=omega,
            alpha=alpha,
            beta=beta,
            ar1=ar1,
            start_price=px0,
        )
        if ticker == "SYNC":
            # SYNC deliberately has holes, so the calendar-alignment and
            # forward-fill logic in data.py has something to chew on.
            drop = frame.index[[37, 38, 150, 400, 401, 402]]
            frame = frame.drop(index=drop)
        out = FIXTURE_DIR / f"{ticker}.parquet"
        frame.to_parquet(out, engine="pyarrow", compression="snappy")
        span = f"{frame.index[0].date()} -> {frame.index[-1].date()}"
        print(f"{ticker}: {len(frame)} rows {span} -> {out}")


if __name__ == "__main__":
    main()
