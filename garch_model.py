"""
garch_model.py
GARCH(1,1) volatility modeling, an upgrade over plain GBM's constant-volatility
assumption.

Plain GBM assumes sigma is constant for the entire forecast horizon. Real
markets exhibit "volatility clustering": calm periods and turbulent periods
each persist for a while (large moves tend to follow large moves). GARCH(1,1)
models today's variance as a function of:
    - a long-run baseline variance (omega)
    - yesterday's squared return (the "ARCH" term — recent shocks)
    - yesterday's variance (the "GARCH" term — volatility persistence)

    sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2

We fit this to historical returns, then forecast a full PATH of future daily
volatilities (not just one constant number) and feed that time-varying
volatility into the Monte Carlo simulation instead of a flat sigma.
"""

import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(log_returns: pd.Series):
    """
    Fit a GARCH(1,1) model to daily log returns.
    Returns are rescaled by 100 (standard practice for the `arch` package —
    it improves numerical optimization stability) and rescaled back afterward.
    """
    scaled_returns = log_returns * 100
    model = arch_model(scaled_returns, vol="Garch", p=1, q=1, dist="normal", mean="Constant")
    fitted = model.fit(disp="off")
    return fitted


def forecast_garch_volatility(fitted_model, n_days: int) -> np.ndarray:
    """
    Forecast daily volatility (sigma, NOT variance, and NOT scaled by 100)
    for each of the next n_days trading days.

    Returns an array of length n_days: one sigma estimate per future day.
    """
    forecast = fitted_model.forecast(horizon=n_days, reindex=False)
    variance_forecast = forecast.variance.values[-1]  # shape (n_days,), still scaled by 100^2
    sigma_forecast = np.sqrt(variance_forecast) / 100  # undo the x100 scaling
    return sigma_forecast


def simulate_gbm_paths_garch(
    s0: float,
    mu_daily: float,
    sigma_path: np.ndarray,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Same GBM simulation as monte_carlo.simulate_gbm_paths, but sigma varies
    by day according to sigma_path (from forecast_garch_volatility) instead
    of being held constant. This lets near-term forecasts reflect current
    volatility regime (e.g. right after an earnings shock) rather than a
    long-run historical average.
    """
    n_days = len(sigma_path)
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    z = rng.standard_normal(size=(n_simulations, n_days))
    # drift correction uses the DAY-SPECIFIC sigma
    daily_log_returns = (mu_daily - 0.5 * sigma_path**2) + sigma_path * z

    cum_log_returns = np.cumsum(daily_log_returns, axis=1)
    price_paths = s0 * np.exp(cum_log_returns)

    paths = np.hstack([np.full((n_simulations, 1), s0), price_paths])
    return paths


def compare_constant_vs_garch_vol(log_returns: pd.Series, n_days: int) -> dict:
    """
    Convenience function: fits GARCH, forecasts the volatility path, and
    compares it to the naive constant-volatility assumption GBM normally
    uses. Useful for a "here's why this matters" chart/table in your report.
    """
    constant_sigma = log_returns.std()

    fitted = fit_garch(log_returns)
    garch_sigma_path = forecast_garch_volatility(fitted, n_days)

    return {
        "constant_sigma_daily": constant_sigma,
        "garch_sigma_path": garch_sigma_path,
        "garch_sigma_day1": garch_sigma_path[0],
        "garch_sigma_final_day": garch_sigma_path[-1],
        "garch_model_summary": fitted.summary(),
    }
