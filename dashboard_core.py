"""
dashboard_core.py
Cached data/compute functions and Plotly chart builders used by app.py.
Kept separate from app.py's page layout so this logic can be unit tested
without needing a running Streamlit session.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import fetch_price_history, compute_log_returns, compute_gbm_parameters
from monte_carlo import simulate_gbm_paths
from garch_model import fit_garch, forecast_garch_volatility, simulate_gbm_paths_garch
from backtest import walk_forward_backtest, compute_coverage


# ---------------------------------------------------------------------------
# Cached data / compute layer
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_data(ticker: str, period: str):
    df = fetch_price_history(ticker, period=period)
    log_returns = compute_log_returns(df)
    params = compute_gbm_parameters(log_returns)
    s0 = float(df["Close"].iloc[-1])
    return df, log_returns, params, s0


@st.cache_data(show_spinner=False, ttl=3600)
def run_gbm_simulation(s0, mu_daily, sigma_daily, n_days, n_sims):
    return simulate_gbm_paths(s0, mu_daily, sigma_daily, n_days, n_sims)


@st.cache_data(show_spinner=False, ttl=3600)
def run_garch(_log_returns, n_days, s0, mu_daily, n_sims):
    # Leading underscore on log_returns tells st.cache_data not to hash it
    # (a pandas Series isn't cheaply hashable); the other scalar args are
    # what determine whether this needs recomputing.
    fitted = fit_garch(_log_returns)
    sigma_path = forecast_garch_volatility(fitted, n_days)
    paths = simulate_gbm_paths_garch(s0, mu_daily, sigma_path, n_sims)
    return sigma_path, paths


@st.cache_data(show_spinner=False, ttl=3600)
def run_backtest(_price_series, _log_returns, horizon_days, lookback_days, step_days):
    bt = walk_forward_backtest(_price_series, _log_returns, horizon_days, lookback_days, step_days)
    cov = compute_coverage(bt) if len(bt) > 0 else pd.DataFrame()
    return bt, cov


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def fan_chart(paths: np.ndarray, ticker: str, n_sample_paths: int = 60):
    days = np.arange(paths.shape[1])
    p5, p25, p50, p75, p95 = (np.percentile(paths, q, axis=0) for q in (5, 25, 50, 75, 95))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=p5, fill="tonexty", fillcolor="rgba(70,130,180,0.15)",
                              line=dict(width=0), name="5th–95th percentile"))
    fig.add_trace(go.Scatter(x=days, y=p75, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=p25, fill="tonexty", fillcolor="rgba(70,130,180,0.35)",
                              line=dict(width=0), name="25th–75th percentile"))
    fig.add_trace(go.Scatter(x=days, y=p50, line=dict(color="navy", width=2.5), name="Median path"))

    sample_idx = np.random.choice(paths.shape[0], size=min(n_sample_paths, paths.shape[0]), replace=False)
    for i in sample_idx:
        fig.add_trace(go.Scatter(x=days, y=paths[i], line=dict(color="rgba(70,130,180,0.06)", width=1),
                                  showlegend=False, hoverinfo="skip"))

    fig.update_layout(title=f"{ticker}: Monte Carlo Simulated Price Paths",
                       xaxis_title="Trading days ahead", yaxis_title="Price ($)",
                       height=450, margin=dict(t=50, b=40))
    return fig


def drawdown_chart(price_series: pd.Series, ticker: str):
    running_max = price_series.cummax()
    drawdown = (price_series - running_max) / running_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy",
                              line=dict(color="firebrick", width=1), fillcolor="rgba(178,34,34,0.35)"))
    fig.update_layout(title=f"{ticker}: Drawdown from Running Peak",
                       yaxis_title="Drawdown (%)", height=350, margin=dict(t=50, b=40))
    return fig


def garch_vs_constant_chart(garch_sigma_path: np.ndarray, constant_sigma: float, ticker: str):
    days = np.arange(1, len(garch_sigma_path) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=garch_sigma_path, line=dict(color="crimson", width=2.5),
                              name="GARCH(1,1) forecast"))
    fig.add_trace(go.Scatter(x=days, y=[constant_sigma] * len(days), line=dict(color="steelblue", width=2, dash="dash"),
                              name="Constant-vol assumption"))
    fig.update_layout(title=f"{ticker}: GARCH Volatility Forecast vs. Constant-Volatility Assumption",
                       xaxis_title="Trading days ahead", yaxis_title="Daily volatility (sigma)",
                       height=400, margin=dict(t=50, b=40))
    return fig


def calibration_chart(backtest_df: pd.DataFrame, coverage_df: pd.DataFrame, ticker: str):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Calibration (PIT) Histogram", "Nominal vs. Empirical Coverage"))

    fig.add_trace(go.Histogram(x=backtest_df["percentile_rank"], nbinsx=20, marker_color="teal",
                                name="Backtest windows"), row=1, col=1)
    expected_height = len(backtest_df) / 20
    fig.add_hline(y=expected_height, line_dash="dash", line_color="black", row=1, col=1)

    fig.add_trace(go.Bar(x=[f"{c*100:.0f}%" for c in coverage_df["nominal_confidence"]],
                          y=coverage_df["nominal_confidence"] * 100, name="Nominal", marker_color="steelblue"),
                  row=1, col=2)
    fig.add_trace(go.Bar(x=[f"{c*100:.0f}%" for c in coverage_df["nominal_confidence"]],
                          y=coverage_df["empirical_coverage"] * 100, name="Empirical", marker_color="darkorange"),
                  row=1, col=2)

    fig.update_layout(height=420, margin=dict(t=60, b=40), barmode="group")
    fig.update_xaxes(title_text="Percentile rank of actual outcome", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Coverage (%)", row=1, col=2)
    return fig
