"""
risk_metrics.py
Core risk metrics used by equity/portfolio risk desks:

- Value at Risk (VaR), computed three ways:
    1. Historical  — empirical percentile of actual past returns
    2. Parametric  — assumes returns are normally distributed (variance-covariance method)
    3. Monte Carlo — empirical percentile of simulated future returns (from monte_carlo.py)
- Conditional VaR / Expected Shortfall — average loss GIVEN that VaR is breached
- Sharpe ratio — risk-adjusted return using total volatility
- Sortino ratio — risk-adjusted return using only downside volatility
- Max drawdown — largest peak-to-trough decline

All VaR/CVaR figures are reported as POSITIVE numbers representing a loss
(the convention used on trading desks), at a given confidence level
(e.g. 95% VaR = the loss that should only be exceeded 5% of the time).
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------

def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: directly take the empirical (1-confidence) percentile
    of past returns. Makes no distributional assumption, just uses what
    actually happened. Weakness: limited by the specific history sampled.
    """
    alpha = 1 - confidence
    var_return = np.percentile(returns, alpha * 100)
    return -var_return  # flip sign so a loss is reported as positive


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (variance-covariance) VaR: assumes returns ~ Normal(mu, sigma).
    VaR = -(mu + sigma * z_alpha), where z_alpha is the alpha-quantile of
    the standard normal (a negative number for alpha < 0.5).
    Weakness: real returns have fatter tails than a normal distribution,
    so this tends to UNDERSTATE tail risk.
    """
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    var_return = mu + sigma * z
    return -var_return


def monte_carlo_var(simulated_paths: np.ndarray, s0: float, confidence: float = 0.95) -> float:
    """
    Monte Carlo VaR: use the distribution of SIMULATED final prices
    (from monte_carlo.simulate_gbm_paths) to compute the empirical
    percentile of simulated returns. Reflects whatever model (GBM, GBM+GARCH,
    etc.) generated the paths, only as good as that model's assumptions.
    """
    final_prices = simulated_paths[:, -1]
    simulated_returns = final_prices / s0 - 1
    alpha = 1 - confidence
    var_return = np.percentile(simulated_returns, alpha * 100)
    return -var_return


# ---------------------------------------------------------------------------
# Conditional VaR / Expected Shortfall
# ---------------------------------------------------------------------------

def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Expected Shortfall: the AVERAGE return in the worst (1-confidence) tail
    of historical returns. Answers "if things go bad, how bad on average?",
    a more informative tail-risk number than VaR alone, since VaR only marks
    the threshold and says nothing about severity beyond it.
    """
    alpha = 1 - confidence
    threshold = np.percentile(returns, alpha * 100)
    tail_returns = returns[returns <= threshold]
    return -tail_returns.mean()


def monte_carlo_cvar(simulated_paths: np.ndarray, s0: float, confidence: float = 0.95) -> float:
    """Same idea as historical_cvar, but over simulated future returns."""
    final_prices = simulated_paths[:, -1]
    simulated_returns = final_prices / s0 - 1
    alpha = 1 - confidence
    threshold = np.percentile(simulated_returns, alpha * 100)
    tail_returns = simulated_returns[simulated_returns <= threshold]
    return -tail_returns.mean()


# ---------------------------------------------------------------------------
# Risk-adjusted return
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.04, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio: (mean excess return / std of returns), annualized.
    Penalizes ALL volatility, upside and downside alike.
    """
    rf_daily = risk_free_annual / periods_per_year
    excess_returns = returns - rf_daily
    if excess_returns.std() == 0:
        return np.nan
    daily_sharpe = excess_returns.mean() / excess_returns.std()
    return daily_sharpe * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.04, periods_per_year: int = 252) -> float:
    """
    Annualized Sortino ratio: like Sharpe, but only penalizes DOWNSIDE
    volatility (returns below the risk-free rate). Two stocks with identical
    Sharpe ratios can have very different Sortino ratios if one's volatility
    is mostly upside surprises and the other's is mostly downside.
    """
    rf_daily = risk_free_annual / periods_per_year
    excess_returns = returns - rf_daily
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    daily_sortino = excess_returns.mean() / downside_std
    return daily_sortino * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def max_drawdown(price_series: pd.Series) -> dict:
    """
    Max drawdown: the largest percentage decline from a running peak.
    Returns the magnitude plus the dates it occurred, since "how bad, and
    when" is what actually matters for a risk write-up.
    """
    running_max = price_series.cummax()
    drawdown = (price_series - running_max) / running_max

    trough_date = drawdown.idxmin()
    max_dd = drawdown.min()
    # peak is the running max as of the trough date
    peak_date = price_series.loc[:trough_date].idxmax()

    return {
        "max_drawdown_pct": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "peak_price": price_series.loc[peak_date],
        "trough_price": price_series.loc[trough_date],
    }


# ---------------------------------------------------------------------------
# Convenience: build a full risk report
# ---------------------------------------------------------------------------

def build_risk_report(
    returns: pd.Series,
    price_series: pd.Series,
    simulated_paths: np.ndarray,
    s0: float,
    confidence: float = 0.95,
    risk_free_annual: float = 0.04,
) -> dict:
    """Bundle every metric above into one dict, ready to print or serialize."""
    dd = max_drawdown(price_series)
    return {
        "confidence": confidence,
        "var_historical": historical_var(returns, confidence),
        "var_parametric": parametric_var(returns, confidence),
        "var_monte_carlo": monte_carlo_var(simulated_paths, s0, confidence),
        "cvar_historical": historical_cvar(returns, confidence),
        "cvar_monte_carlo": monte_carlo_cvar(simulated_paths, s0, confidence),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_annual),
        "sortino_ratio": sortino_ratio(returns, risk_free_annual),
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "max_drawdown_peak_date": dd["peak_date"],
        "max_drawdown_trough_date": dd["trough_date"],
    }
