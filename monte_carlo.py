"""
monte_carlo.py
Monte Carlo simulation of future stock prices using Geometric Brownian Motion (GBM).

GBM assumes:
    dS = mu * S * dt + sigma * S * dW

Discretized (what we actually simulate), for each time step dt:
    S_t+1 = S_t * exp( (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z )

where Z ~ N(0, 1) is a random standard normal draw each step.
The (mu - 0.5*sigma^2) term is the "drift correction", it accounts for the
fact that log returns are being converted back to price levels
(Ito's lemma / Jensen's inequality correction).
"""

import numpy as np


def simulate_gbm_paths(
    s0: float,
    mu_daily: float,
    sigma_daily: float,
    n_days: int,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate n_simulations price paths over n_days trading days.

    Returns:
        np.ndarray of shape (n_simulations, n_days + 1)
        Column 0 is s0 for every path; each subsequent column is one more
        simulated trading day forward.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = 1  # daily steps, since mu/sigma are already daily
    drift = mu_daily - 0.5 * sigma_daily**2

    # Random shocks: shape (n_simulations, n_days)
    z = rng.standard_normal(size=(n_simulations, n_days))
    daily_log_returns = drift * dt + sigma_daily * np.sqrt(dt) * z

    # Cumulative log return path, then exponentiate to get price path
    cum_log_returns = np.cumsum(daily_log_returns, axis=1)
    price_paths = s0 * np.exp(cum_log_returns)

    # Prepend the starting price s0 as column 0
    paths = np.hstack([np.full((n_simulations, 1), s0), price_paths])
    return paths


def summarize_final_prices(paths: np.ndarray) -> dict:
    """Summary stats for the final simulated day's price distribution."""
    final_prices = paths[:, -1]
    return {
        "mean": np.mean(final_prices),
        "median": np.median(final_prices),
        "std": np.std(final_prices),
        "p5": np.percentile(final_prices, 5),
        "p25": np.percentile(final_prices, 25),
        "p75": np.percentile(final_prices, 75),
        "p95": np.percentile(final_prices, 95),
        "min": np.min(final_prices),
        "max": np.max(final_prices),
    }


def probability_above(paths: np.ndarray, threshold: float) -> float:
    """P(final simulated price > threshold), based on simulated paths."""
    final_prices = paths[:, -1]
    return float(np.mean(final_prices > threshold))


def probability_below(paths: np.ndarray, threshold: float) -> float:
    """P(final simulated price < threshold)."""
    final_prices = paths[:, -1]
    return float(np.mean(final_prices < threshold))


def probability_in_range(paths: np.ndarray, low: float, high: float) -> float:
    """P(low < final simulated price < high)."""
    final_prices = paths[:, -1]
    return float(np.mean((final_prices > low) & (final_prices < high)))
