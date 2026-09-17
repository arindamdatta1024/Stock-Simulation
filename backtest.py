"""
backtest.py

The core question answers: when the model says "there's a 90% chance
the price lands in this range," is that actually true historically? A model
can look sophisticated and still be badly calibrated, this is where you
find out.

Method: walk forward through history in a rolling window. At each point,
fit GBM parameters using only data available UP TO that point (no lookahead),
then check where the ACTUAL price `horizon_days` later falls within the
model's predicted distribution.

Two standard forecast-verification techniques are used:

1. PIT (Probability Integral Transform) histogram - for each backtest window,
   compute the percentile rank of the actual outcome under the model's
   predicted distribution. If the model is well-calibrated, these percentile
   ranks should be UNIFORMLY distributed between 0 and 1 across many windows.
   A histogram that's skewed, U-shaped, or peaked reveals specific flaws
   (e.g. a U-shape means the model's intervals are too narrow, reality
   lands in the tails more often than predicted).

2. Coverage comparison — for a stated confidence level (e.g. 90%), what
   fraction of backtest windows actually had the real outcome fall inside
   the model's 90% interval? It should be close to 90%. Consistently lower
   means the model understates risk (overconfident); consistently higher
   means it overstates risk (too conservative).

Note on method: rather than re-running a full Monte Carlo simulation at
every single backtest window (slow, and unnecessary here), this uses GBM's
known analytical property: under GBM, the SUM of daily log returns over
`horizon_days` is itself approximately Normal(horizon*mu, sigma*sqrt(horizon)).
That lets us compute exact percentile ranks and interval bounds directly from
the normal distribution instead of simulating thousands of paths per window.
"""

import numpy as np
import pandas as pd
from scipy import stats


def walk_forward_backtest(
    price_series: pd.Series,
    log_returns: pd.Series,
    horizon_days: int = 30,
    lookback_days: int = 252,
    step_days: int = 5,
) -> pd.DataFrame:
    """
    Roll forward through history. At each step:
      - Fit mu/sigma using only the `lookback_days` of returns immediately
        prior (no lookahead bias).
      - Predict the distribution of the log return over the next
        `horizon_days`.
      - Compare against what ACTUALLY happened.

    Returns a DataFrame with one row per backtest window.
    """
    records = []
    n = len(price_series)

    # i is the index of the "today" we're forecasting FROM
    start = lookback_days
    end = n - horizon_days

    for i in range(start, end, step_days):
        train_returns = log_returns.iloc[i - lookback_days:i]
        mu = train_returns.mean()
        sigma = train_returns.std()

        s0 = price_series.iloc[i]
        s_future = price_series.iloc[i + horizon_days]
        actual_log_return = np.log(s_future / s0)

        predicted_mean = mu * horizon_days
        predicted_std = sigma * np.sqrt(horizon_days)

        if predicted_std == 0 or np.isnan(predicted_std):
            continue

        # Percentile rank of the actual outcome under the model's predicted
        # distribution (this is the PIT value for this window)
        percentile_rank = stats.norm.cdf(actual_log_return, loc=predicted_mean, scale=predicted_std)

        records.append({
            "forecast_date": price_series.index[i],
            "target_date": price_series.index[i + horizon_days],
            "s0": s0,
            "s_actual": s_future,
            "predicted_mean_return": predicted_mean,
            "predicted_std_return": predicted_std,
            "actual_log_return": actual_log_return,
            "percentile_rank": percentile_rank,
        })

    return pd.DataFrame(records)


def compute_coverage(backtest_df: pd.DataFrame, confidence_levels=(0.50, 0.80, 0.90, 0.95)) -> pd.DataFrame:
    """
    For each confidence level, compute what fraction of backtest windows had
    the actual outcome fall inside the model's predicted interval at that
    confidence level, and compare to the nominal (stated) level.

    A well-calibrated model has empirical coverage close to nominal coverage.
    """
    rows = []
    for conf in confidence_levels:
        alpha = 1 - conf
        lower_pct = alpha / 2
        upper_pct = 1 - alpha / 2
        within = (backtest_df["percentile_rank"] >= lower_pct) & (backtest_df["percentile_rank"] <= upper_pct)
        empirical_coverage = within.mean()
        rows.append({
            "nominal_confidence": conf,
            "empirical_coverage": empirical_coverage,
            "difference": empirical_coverage - conf,
        })
    return pd.DataFrame(rows)


def calibration_summary_text(coverage_df: pd.DataFrame) -> str:
    """Human-readable verdict on calibration quality."""
    lines = []
    for _, row in coverage_df.iterrows():
        nominal = row["nominal_confidence"]
        empirical = row["empirical_coverage"]
        diff = row["difference"]
        verdict = "well-calibrated"
        if diff < -0.07:
            verdict = "OVERCONFIDENT (real outcomes land outside the band more than expected)"
        elif diff > 0.07:
            verdict = "too conservative (band is wider than necessary)"
        lines.append(
            f"  {nominal*100:.0f}% interval -> actually captured outcome {empirical*100:.1f}% of the time  [{verdict}]"
        )
    return "\n".join(lines)
