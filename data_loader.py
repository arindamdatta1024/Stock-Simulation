"""
data_loader.py
Fetches historical price data and computes the statistics needed
to drive a Geometric Brownian Motion (GBM) simulation.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download historical OHLCV data for a ticker.

    period examples: '1y', '2y', '5y', '10y', 'max'
    """
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten if so
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def compute_log_returns(price_df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Compute daily log returns: ln(P_t / P_t-1)
    Log returns are used (not simple % returns) because they are additive
    over time and are the standard basis for GBM.
    """
    prices = price_df[price_col]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def compute_gbm_parameters(log_returns: pd.Series) -> dict:
    """
    Derive the two parameters GBM needs from historical daily log returns:

    - mu (drift): average daily log return
    - sigma (volatility): standard deviation of daily log returns

    Also returns annualized versions (x252 trading days) since that's how
    you'll usually discuss/report these numbers.
    """
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()

    return {
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "mu_annual": mu_daily * 252,
        "sigma_annual": sigma_daily * np.sqrt(252),
    }


if __name__ == "__main__":
    # Quick manual test
    ticker = "AAPL"
    df = fetch_price_history(ticker, period="5y")
    returns = compute_log_returns(df)
    params = compute_gbm_parameters(returns)

    print(f"\n{ticker} — {len(df)} trading days loaded")
    print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")
    print(f"Daily drift (mu):      {params['mu_daily']:.6f}")
    print(f"Daily volatility (sigma): {params['sigma_daily']:.6f}")
    print(f"Annualized drift:      {params['mu_annual']:.4f} ({params['mu_annual']*100:.2f}%)")
    print(f"Annualized volatility: {params['sigma_annual']:.4f} ({params['sigma_annual']*100:.2f}%)")
