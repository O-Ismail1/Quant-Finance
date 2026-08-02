# Statistical Arbitrage Pairs Trading

A market-neutral trading strategy based on identifying pairs of stocks with a stable long-term relationship.

## Main concepts:

- Correlation filtering
- Engle-Granger cointegration testing
- Regression-based hedge ratio estimation
- Mean reversion signals
- Out-of-sample backtesting

## Methodology:

- Candidate pairs selected using historical correlation
- Cointegration tested using Augmented Dickey-Fuller test
- Rolling hedge ratios estimated using OLS regression
- Trading signals generated using spread z-score
