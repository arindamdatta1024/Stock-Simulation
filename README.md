# Stock Forecasting & Risk Dashboard

I built this to teach myself how quant/risk analysts actually think about
uncertainty, while combining my previous math knowledge from courses I took such as stochastic processes and time series analysis. It takes a stock ticker, simulates thousands of possible
future price paths, and tells you things like "there's a 62% chance this
stock is above $210 in 30 days", along with the risk numbers (VaR, CVaR,
Sharpe/Sortino, drawdown) that a risk desk would actually care about.

It's not a stock picker and it won't tell you what to buy. It's a way of
being honest about what we don't know, instead of pretending a single price
target means anything.

There's a live dashboard (Streamlit) and a command-line version that prints
a full report and saves the charts as PNGs. Both use the same underlying
code.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

That opens the dashboard in your browser. Pick a ticker in the sidebar and
poke around, there are four tabs (forecast, risk metrics, GARCH
volatility, backtest).

If you'd rather just get a printed report and some chart PNGs without the
browser:

```bash
python main.py --ticker AAPL --days 30 --sims 10000
```

## The idea, in plain terms

Stock prices don't move in a straight line, they wander, roughly
randomly, with some average drift and some amount of daily jitter
(volatility). If you know a stock's historical average return and how
choppy it's been, you can simulate thousands of "what if" versions of the
next 30 days and see where they end up. That's Monte Carlo simulation. Run
it 10,000 times and you get a whole distribution of outcomes instead of one
guess, which is a much more honest way to talk about the future.

The model underneath is Geometric Brownian Motion (GBM), which is the same
math finance people use as the starting point for option pricing (it's the
backbone of Black-Scholes). It's not the fanciest model out there, but it's
the right place to start, and its assumptions are simple enough that you
can reason about exactly where it breaks down, which matters more than
people think.

## What's actually in here

- `data_loader.py` — pulls price history off Yahoo Finance and turns it
  into the two numbers GBM needs: average daily return (drift) and how
  volatile the stock's been (sigma).
- `monte_carlo.py` — the actual simulation. Simulates a bunch of price
  paths and lets you ask questions like "what's the probability this ends
  up above $X."
- `risk_metrics.py` — Value at Risk, three different ways (historical,
  parametric, and Monte Carlo-based), plus Conditional VaR, Sharpe/Sortino
  ratios, and max drawdown. I go into why there are three VaR methods
  below, because that tripped me up at first.
- `garch_model.py` — a GARCH(1,1) model, which is basically a smarter
  volatility estimate. Plain GBM assumes volatility is constant, which
  isn't true, markets get choppy and then calm down in clusters. GARCH
  picks up on that.
- `backtest.py` — walks back through history and checks whether the
  model's confidence intervals were actually right. This was the most
  interesting part to build, honestly, because it's the part that tells
  you whether to trust anything else in this repo.
- `dashboard_core.py` / `app.py` — the Streamlit dashboard. Split into two
  files so the actual logic (caching, chart building) isn't tangled up
  with the page layout, makes it easier to test and easier to read.

## On the risk numbers

**VaR (Value at Risk)** answers "how much could I lose, with X% confidence,
over some time period." I calculate it three different ways because they
each have a different failure mode and it's worth knowing all three:

- Historical VaR just looks at what actually happened in the past. No
  assumptions, but it's limited by whatever history you happened to sample.
- Parametric VaR assumes returns follow a normal distribution and does the
  math from there. Fast, but real markets have fatter tails than a normal
  distribution predicts, so this one tends to make things look safer than
  they are.
- Monte Carlo VaR comes straight out of the simulated price paths.

**CVaR** (also called Expected Shortfall) is the number I actually trust
more than VaR. VaR tells you the threshold; CVaR tells you how bad things
get on average once you're past that threshold. If someone asks "sure, but
how bad is bad," CVaR is the answer.

**Sharpe and Sortino** are both risk-adjusted return measures. Sharpe
penalizes volatility in either direction, which always felt a little off
to me, nobody's upset about upside volatility. Sortino only penalizes
downside volatility, which is closer to how people actually think about
risk.

## The GARCH detour

This was the part of the project where I actually learned something I
didn't expect. Plain GBM treats volatility as one flat number for the
whole forecast, as if a stock is exactly as jumpy on day 30 as it is
today. That's obviously not how markets work; volatility comes in waves.
GARCH(1,1) models tomorrow's volatility as a function of a long-run
average, yesterday's shock, and yesterday's volatility, so it can pick up
on "hey, things have been jumpy lately, that'll probably continue for a
bit before settling down."

The dashboard shows both side by side, and lets you see how much the VaR
estimate shifts once you stop assuming flat volatility. Sometimes it
barely moves. Sometimes it moves a lot, depending on what the stock's been
doing lately.

## Does the model actually work, though?

This is the question I think most similar projects skip, and it's the one
that matters most. A model can spit out a confident-looking "90% chance"
range and just be wrong.

`backtest.py` checks this directly: it goes back through history, and at
each point pretends it's "today," fits the model using only the data that
would've been available at that point (no cheating by looking ahead), then
checks where the actual price landed relative to what the model predicted.
Do that a couple hundred times across history and you can ask: when the
model says "90% confidence interval," does reality actually land inside
that range about 90% of the time?

On the tickers I tested, it came out reasonably well calibrated, which was
a relief. It's not perfect, real markets have fatter tails than GBM
assumes, so I'd expect the model to slightly understate how often things
go really wrong. One honest caveat: because I space the backtest windows
closer together than the forecast horizon, consecutive windows overlap and
aren't fully independent, which makes the results look a little tighter
than they truly are. If I were doing this more rigorously I'd space the
windows out further apart, at the cost of having way fewer of them to test
against.

## Deploying it

I put this on Streamlit Community Cloud (free) so I could link a live demo
instead of just a GitHub repo:

1. Push the repo to GitHub.
2. Go to share.streamlit.io, sign in with GitHub.
3. New app → pick the repo → set the main file to `app.py`.
4. It installs everything from `requirements.txt` automatically and gives
   you a public URL.

Having something clickable instead of just a repo link made a real
difference in how it felt to talk about, it's a lot easier to say "here,
try it" than to walk someone through code.

## What I'd add if I kept going

- Extend this to a portfolio of stocks instead of one at a time, would
  need a covariance matrix and probably the efficient frontier stuff from
  Modern Portfolio Theory.
- A jump-diffusion model, to handle the "stock drops 12% overnight on
  earnings" case that GBM just can't represent.
- Compare against a couple of dumb baselines (like "assume tomorrow equals
  today") just to make sure the fancier models are actually earning their
  complexity.

## One honest limitation

Even with GARCH layered in, this is still built on the assumption that
returns are roughly log-normal. Real markets have fatter tails and
occasional sharp jumps that this kind of model structurally can't see
coming. I think that's worth saying out loud rather than burying, the
backtest is exactly the tool that would catch it if it became a real
problem, and it's the honest answer if anyone asks "what's this model bad
at."
