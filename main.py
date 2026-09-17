"""
main.py
Full pipeline: data -> GBM Monte Carlo simulation -> risk analytics ->
GARCH volatility comparison -> walk-forward backtesting / calibration check.

Run:
    python main.py --ticker AAPL --days 30 --sims 10000

Produces:
    - Console report: simulation summary, probability statements, VaR/CVaR
      (3 methods), Sharpe/Sortino, max drawdown, GARCH vs constant-vol
      comparison, backtest calibration table
    - {ticker}_fan_chart.png             — simulated price paths
    - {ticker}_garch_vs_constant_vol.png — volatility forecast comparison
    - {ticker}_drawdown.png              — historical drawdown
    - {ticker}_calibration.png           — backtest calibration (PIT histogram + coverage)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_loader import fetch_price_history, compute_log_returns, compute_gbm_parameters
from monte_carlo import simulate_gbm_paths, summarize_final_prices, probability_above
from risk_metrics import build_risk_report, monte_carlo_var, monte_carlo_cvar
from garch_model import fit_garch, forecast_garch_volatility, simulate_gbm_paths_garch
from backtest import walk_forward_backtest, compute_coverage, calibration_summary_text
from pathlib import Path


def run(
    ticker: str,
    forecast_days: int,
    n_simulations: int,
    history_period: str = "5y",
    confidence: float = 0.95,
    run_backtest: bool = True,
    backtest_lookback: int = 252,
    backtest_step: int = 5,
):
    print(f"\n{'='*70}")
    print(f"  {ticker} — Probabilistic Forecast & Risk Report")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 1. Data + GBM parameters
    # ------------------------------------------------------------------
    df = fetch_price_history(ticker, period=history_period)
    log_returns = compute_log_returns(df)
    params = compute_gbm_parameters(log_returns)
    s0 = float(df["Close"].iloc[-1])
    price_series = df["Close"]

    print(f"\nCurrent price: ${s0:.2f}   |   History: {len(df)} days ({history_period})")
    print(f"Daily drift (mu):          {params['mu_daily']:.6f}")
    print(f"Daily volatility (sigma):  {params['sigma_daily']:.6f}")
    print(f"Annualized volatility:     {params['sigma_annual']*100:.2f}%")

    # ------------------------------------------------------------------
    # 2. Monte Carlo simulation (constant-volatility GBM) + fan chart
    # ------------------------------------------------------------------
    paths = simulate_gbm_paths(
        s0=s0, mu_daily=params["mu_daily"], sigma_daily=params["sigma_daily"],
        n_days=forecast_days, n_simulations=n_simulations,
    )
    summary = summarize_final_prices(paths)

    print(f"\n--- Simulated price distribution after {forecast_days} trading days ---")
    print(f"  Mean: ${summary['mean']:.2f}   Median: ${summary['median']:.2f}   Std: ${summary['std']:.2f}")
    print(f"  5th pct: ${summary['p5']:.2f}   |  95th pct: ${summary['p95']:.2f}")

    print(f"\n--- Example probability statements ---")
    for pct in [-0.10, -0.05, 0.0, 0.05, 0.10]:
        threshold = s0 * (1 + pct)
        p = probability_above(paths, threshold)
        print(f"  P(price above ${threshold:.2f}, {pct*100:+.0f}%): {p*100:5.1f}%")

    plot_fan_chart(paths, ticker, forecast_days)

    # ------------------------------------------------------------------
    # 3. Risk report: VaR/CVaR (3 methods), Sharpe, Sortino, max drawdown
    # ------------------------------------------------------------------
    report = build_risk_report(
        returns=log_returns, price_series=price_series,
        simulated_paths=paths, s0=s0, confidence=confidence,
    )

    print(f"\n--- Value at Risk (1-day, {confidence*100:.0f}% confidence) ---")
    print(f"  Historical VaR:   {report['var_historical']*100:6.2f}%")
    print(f"  Parametric VaR:   {report['var_parametric']*100:6.2f}%")
    print(f"  Monte Carlo VaR:  {report['var_monte_carlo']*100:6.2f}%  [{forecast_days}-day horizon]")

    print(f"\n--- Conditional VaR / Expected Shortfall ---")
    print(f"  Historical CVaR:   {report['cvar_historical']*100:6.2f}%")
    print(f"  Monte Carlo CVaR:  {report['cvar_monte_carlo']*100:6.2f}%  [{forecast_days}-day horizon]")

    print(f"\n--- Risk-Adjusted Return ---")
    print(f"  Sharpe ratio (annualized):  {report['sharpe_ratio']:.3f}")
    print(f"  Sortino ratio (annualized): {report['sortino_ratio']:.3f}")

    print(f"\n--- Max Drawdown ---")
    print(f"  Max drawdown: {report['max_drawdown_pct']*100:.2f}%")
    print(f"  Peak:   {report['max_drawdown_peak_date'].date()}   Trough: {report['max_drawdown_trough_date'].date()}")

    plot_drawdown(price_series, ticker)

    # ------------------------------------------------------------------
    # 4. GARCH volatility forecast vs. constant-vol assumption
    # ------------------------------------------------------------------
    print(f"\n--- GARCH(1,1) Volatility Forecast vs. Constant-Vol Assumption ---")
    fitted_garch = fit_garch(log_returns)
    garch_sigma_path = forecast_garch_volatility(fitted_garch, forecast_days)
    constant_sigma = params["sigma_daily"]

    print(f"  Constant-vol (flat):    {constant_sigma:.5f}")
    print(f"  GARCH day-1 forecast:   {garch_sigma_path[0]:.5f}")
    print(f"  GARCH day-{forecast_days} forecast: {garch_sigma_path[-1]:.5f}")

    paths_garch = simulate_gbm_paths_garch(
        s0=s0, mu_daily=params["mu_daily"], sigma_path=garch_sigma_path, n_simulations=n_simulations,
    )
    garch_var = monte_carlo_var(paths_garch, s0, confidence)
    garch_cvar = monte_carlo_cvar(paths_garch, s0, confidence)

    print(f"\n  {forecast_days}-day VaR   — constant-vol: {report['var_monte_carlo']*100:.2f}%   |   GARCH-vol: {garch_var*100:.2f}%")
    print(f"  {forecast_days}-day CVaR  — constant-vol: {report['cvar_monte_carlo']*100:.2f}%   |   GARCH-vol: {garch_cvar*100:.2f}%")

    plot_garch_vs_constant(garch_sigma_path, constant_sigma, ticker)

    # ------------------------------------------------------------------
    # 5. Walk-forward backtest / calibration check
    # ------------------------------------------------------------------
    backtest_df, coverage_df = None, None
    if run_backtest:
        min_required = backtest_lookback + forecast_days + backtest_step
        if len(price_series) < min_required:
            print(f"\n--- Backtest skipped: need at least {min_required} days of history, have {len(price_series)} ---")
        else:
            print(f"\n--- Walk-Forward Backtest ({forecast_days}-day horizon, {backtest_lookback}-day lookback) ---")
            backtest_df = walk_forward_backtest(
                price_series, log_returns,
                horizon_days=forecast_days, lookback_days=backtest_lookback, step_days=backtest_step,
            )
            print(f"  {len(backtest_df)} backtest windows evaluated")

            coverage_df = compute_coverage(backtest_df)
            print(f"\n  Is the model's uncertainty band actually trustworthy?")
            print(calibration_summary_text(coverage_df))

            plot_calibration(backtest_df, coverage_df, ticker)

    return {
        "report": report,
        "paths": paths,
        "garch_sigma_path": garch_sigma_path,
        "backtest_df": backtest_df,
        "coverage_df": coverage_df,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fan_chart(paths: np.ndarray, ticker: str, forecast_days: int, n_sample_paths: int = 100):
    days = np.arange(paths.shape[1])
    p5, p25, p50, p75, p95 = (np.percentile(paths, q, axis=0) for q in (5, 25, 50, 75, 95))

    fig, ax = plt.subplots(figsize=(10, 6))
    sample_idx = np.random.choice(paths.shape[0], size=min(n_sample_paths, paths.shape[0]), replace=False)
    for i in sample_idx:
        ax.plot(days, paths[i], color="steelblue", alpha=0.04, linewidth=0.8)

    ax.fill_between(days, p5, p95, color="steelblue", alpha=0.15, label="5th–95th percentile")
    ax.fill_between(days, p25, p75, color="steelblue", alpha=0.30, label="25th–75th percentile")
    ax.plot(days, p50, color="navy", linewidth=2, label="Median path")
    ax.set_title(f"{ticker}: Monte Carlo Simulated Price Paths ({forecast_days} trading days)")
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ticker}_fan_chart.png", dpi=150)
    print(f"\nSaved: {ticker}_fan_chart.png")


def plot_drawdown(price_series, ticker: str):
    running_max = price_series.cummax()
    drawdown = (price_series - running_max) / running_max * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="firebrick", linewidth=0.8)
    ax.set_title(f"{ticker}: Drawdown from Running Peak")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ticker}_drawdown.png", dpi=150)
    print(f"Saved: {ticker}_drawdown.png")


def plot_garch_vs_constant(garch_sigma_path: np.ndarray, constant_sigma: float, ticker: str):
    days = np.arange(1, len(garch_sigma_path) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, garch_sigma_path, color="crimson", linewidth=2, label="GARCH(1,1) forecast")
    ax.axhline(constant_sigma, color="steelblue", linestyle="--", linewidth=2, label="Constant-vol assumption")
    ax.set_title(f"{ticker}: GARCH Volatility Forecast vs. Constant-Volatility Assumption")
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Daily volatility (sigma)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ticker}_garch_vs_constant_vol.png", dpi=150)
    print(f"Saved: {ticker}_garch_vs_constant_vol.png")


def plot_calibration(backtest_df, coverage_df, ticker: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # PIT histogram: should be roughly flat/uniform if well-calibrated
    ax = axes[0]
    ax.hist(backtest_df["percentile_rank"], bins=20, range=(0, 1), color="teal", alpha=0.75, edgecolor="white")
    expected_height = len(backtest_df) / 20
    ax.axhline(expected_height, color="black", linestyle="--", linewidth=1.5, label="Perfectly uniform")
    ax.set_title(f"{ticker}: Calibration (PIT) Histogram")
    ax.set_xlabel("Percentile rank of actual outcome")
    ax.set_ylabel("Count of backtest windows")
    ax.legend()

    # Coverage comparison: nominal vs empirical
    ax = axes[1]
    x = np.arange(len(coverage_df))
    width = 0.35
    ax.bar(x - width/2, coverage_df["nominal_confidence"] * 100, width, label="Nominal (stated)", color="steelblue")
    ax.bar(x + width/2, coverage_df["empirical_coverage"] * 100, width, label="Empirical (actual)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c*100:.0f}%" for c in coverage_df["nominal_confidence"]])
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"{ticker}: Nominal vs. Empirical Interval Coverage")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{ticker}_calibration.png", dpi=150)
    print(f"Saved: {ticker}_calibration.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probabilistic stock forecasting & risk analytics")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--days", type=int, default=30, help="Forecast horizon in trading days")
    parser.add_argument("--sims", type=int, default=10_000, help="Number of Monte Carlo simulations")
    parser.add_argument("--history", type=str, default="5y", help="History window, e.g. 1y, 2y, 5y")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for VaR/CVaR")
    parser.add_argument("--no-backtest", action="store_true", help="Skip the walk-forward backtest")
    parser.add_argument("--backtest-lookback", type=int, default=252, help="Days of history per backtest fit window")
    parser.add_argument("--backtest-step", type=int, default=5, help="Days between backtest windows")
    args = parser.parse_args()

    run(
        ticker=args.ticker,
        forecast_days=args.days,
        n_simulations=args.sims,
        history_period=args.history,
        confidence=args.confidence,
        run_backtest=not args.no_backtest,
        backtest_lookback=args.backtest_lookback,
        backtest_step=args.backtest_step,
    )
