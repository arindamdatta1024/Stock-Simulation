"""
app.py

Run locally:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from monte_carlo import summarize_final_prices, probability_above
from risk_metrics import build_risk_report, monte_carlo_var, monte_carlo_cvar
from backtest import calibration_summary_text
from dashboard_core import (
    load_data, run_gbm_simulation, run_garch, run_backtest,
    fan_chart, drawdown_chart, garch_vs_constant_chart, calibration_chart,
)

st.set_page_config(page_title="Stock Forecast & Risk Analytics", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Forecast Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
history_period = st.sidebar.selectbox("History window", ["1y", "2y", "5y", "10y"], index=2)
forecast_days = st.sidebar.slider("Forecast horizon (trading days)", 5, 120, 30)
n_simulations = st.sidebar.select_slider("Monte Carlo simulations", options=[1000, 2500, 5000, 10000, 25000], value=10000)
confidence = st.sidebar.slider("VaR/CVaR confidence level", 0.80, 0.99, 0.95, step=0.01)

with st.sidebar.expander("Backtest settings"):
    run_bt = st.checkbox("Run walk-forward backtest", value=True)
    bt_lookback = st.slider("Backtest lookback window (days)", 60, 500, 252)
    bt_step = st.slider("Backtest step size (days)", 1, 20, 5)

st.sidebar.caption("Data via Yahoo Finance (yfinance). Educational use only — not investment advice.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("📈 Probabilistic Stock Forecasting & Risk Analytics")
st.caption("GBM Monte Carlo simulation · GARCH volatility · VaR/CVaR risk metrics · walk-forward backtesting")

if not ticker:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

try:
    with st.spinner(f"Loading {ticker} price history..."):
        df, log_returns, params, s0 = load_data(ticker, history_period)
        price_series = df["Close"]
except Exception as e:
    st.error(f"Couldn't load data for '{ticker}'. Check the symbol and try again.\n\nDetails: {e}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${s0:,.2f}")
col2.metric("Annualized Volatility", f"{params['sigma_annual']*100:.1f}%")
col3.metric("Daily Drift (μ)", f"{params['mu_daily']:.5f}")
col4.metric("History Used", f"{len(df)} days")

tab_forecast, tab_risk, tab_garch, tab_backtest = st.tabs(
    ["🎲 Forecast", "⚠️ Risk Metrics", "📊 GARCH Volatility", "✅ Backtest & Calibration"]
)

# --- Forecast tab ---
with tab_forecast:
    with st.spinner("Running Monte Carlo simulation..."):
        paths = run_gbm_simulation(s0, params["mu_daily"], params["sigma_daily"], forecast_days, n_simulations)
        summary = summarize_final_prices(paths)

    st.plotly_chart(fan_chart(paths, ticker), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Median price in {forecast_days}d", f"${summary['median']:.2f}")
    c2.metric("5th percentile", f"${summary['p5']:.2f}")
    c3.metric("95th percentile", f"${summary['p95']:.2f}")

    st.subheader("Probability statements")
    prob_rows = []
    for pct in [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]:
        threshold = s0 * (1 + pct)
        p = probability_above(paths, threshold)
        prob_rows.append({"Move": f"{pct*100:+.0f}%", "Threshold": f"${threshold:.2f}",
                           "P(price above threshold)": f"{p*100:.1f}%"})
    st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)

    custom_price = st.number_input("Or check a specific target price ($)", value=round(s0, 2), step=1.0)
    p_above = probability_above(paths, custom_price)
    st.write(f"**P(price above ${custom_price:.2f} in {forecast_days} trading days): {p_above*100:.1f}%**")

# --- Risk metrics tab ---
with tab_risk:
    report = build_risk_report(log_returns, price_series, paths, s0, confidence)

    st.subheader(f"Value at Risk — {confidence*100:.0f}% confidence")
    var_df = pd.DataFrame([
        {"Method": "Historical (1-day)", "VaR": f"{report['var_historical']*100:.2f}%"},
        {"Method": "Parametric (1-day)", "VaR": f"{report['var_parametric']*100:.2f}%"},
        {"Method": f"Monte Carlo ({forecast_days}-day)", "VaR": f"{report['var_monte_carlo']*100:.2f}%"},
    ])
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    st.subheader("Conditional VaR (Expected Shortfall)")
    cvar_df = pd.DataFrame([
        {"Method": "Historical (1-day)", "CVaR": f"{report['cvar_historical']*100:.2f}%"},
        {"Method": f"Monte Carlo ({forecast_days}-day)", "CVaR": f"{report['cvar_monte_carlo']*100:.2f}%"},
    ])
    st.dataframe(cvar_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("Sharpe ratio (annualized)", f"{report['sharpe_ratio']:.3f}")
    c2.metric("Sortino ratio (annualized)", f"{report['sortino_ratio']:.3f}")

    st.subheader("Max Drawdown")
    st.metric("Max drawdown", f"{report['max_drawdown_pct']*100:.2f}%")
    st.caption(f"Peak: {report['max_drawdown_peak_date'].date()}  →  Trough: {report['max_drawdown_trough_date'].date()}")
    st.plotly_chart(drawdown_chart(price_series, ticker), use_container_width=True)

# --- GARCH tab ---
with tab_garch:
    with st.spinner("Fitting GARCH(1,1) model..."):
        garch_sigma_path, garch_paths = run_garch(log_returns, forecast_days, s0, params["mu_daily"], n_simulations)

    st.plotly_chart(garch_vs_constant_chart(garch_sigma_path, params["sigma_daily"], ticker), use_container_width=True)

    garch_var = monte_carlo_var(garch_paths, s0, confidence)
    garch_cvar = monte_carlo_cvar(garch_paths, s0, confidence)
    const_var = monte_carlo_var(paths, s0, confidence)
    const_cvar = monte_carlo_cvar(paths, s0, confidence)

    compare_df = pd.DataFrame([
        {"Metric": f"{forecast_days}-day VaR", "Constant-vol GBM": f"{const_var*100:.2f}%", "GARCH-vol GBM": f"{garch_var*100:.2f}%"},
        {"Metric": f"{forecast_days}-day CVaR", "Constant-vol GBM": f"{const_cvar*100:.2f}%", "GARCH-vol GBM": f"{garch_cvar*100:.2f}%"},
    ])
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.caption(
        "GARCH(1,1) lets near-term volatility reflect the current regime "
        "(e.g. elevated after a recent shock) instead of assuming a flat "
        "historical average for the entire forecast horizon."
    )

# --- Backtest tab ---
with tab_backtest:
    if not run_bt:
        st.info("Backtest disabled — enable it in the sidebar to see calibration results.")
    else:
        min_required = bt_lookback + forecast_days + bt_step
        if len(price_series) < min_required:
            st.warning(f"Not enough history for this backtest configuration. "
                       f"Need at least {min_required} days, have {len(price_series)}. "
                       f"Try a longer history window or shorter lookback.")
        else:
            with st.spinner("Running walk-forward backtest..."):
                backtest_df, coverage_df = run_backtest(price_series, log_returns, forecast_days, bt_lookback, bt_step)

            st.write(f"**{len(backtest_df)} backtest windows evaluated** "
                     f"({bt_lookback}-day lookback, {forecast_days}-day horizon, stepping every {bt_step} days)")

            st.plotly_chart(calibration_chart(backtest_df, coverage_df, ticker), use_container_width=True)

            st.subheader("Is the model's uncertainty band trustworthy?")
            st.text(calibration_summary_text(coverage_df))

            st.caption(
                "⚠️ With a step size shorter than the forecast horizon, backtest windows overlap "
                "and aren't fully independent — this can make calibration look tighter than it "
                "truly is. Set the step size close to the forecast horizon for a stricter test."
            )

st.divider()
st.caption(
    "Built with GBM Monte Carlo simulation, GARCH(1,1) volatility modeling, and walk-forward "
    "backtesting. Educational project — not financial advice."
)
