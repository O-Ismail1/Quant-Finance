import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as web

# download S&P 500 data
Tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                       storage_options={"User-Agent": "Mozilla/5.0"})[0].Symbol
Tickers = Tickers.str.replace(".", "-", regex=False)

df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                   storage_options={"User-Agent": "Mozilla/5.0"})[1]

# convert date
df[('Effective Date', 'Effective Date')] = pd.to_datetime(df[('Effective Date', 
                                                              'Effective Date')])

# adjust current S&P 500 constituents using membership changes
# from 2020-2026 to reduce survivorship bias
df = df[df[('Effective Date', 'Effective Date')] >= pd.Timestamp("2020-01-01")]
df = df.sort_values(('Effective Date', 'Effective Date'),ascending=False)
Tickers = set(Tickers)

for _, row in df.iterrows():

    added = row[('Added', 'Ticker')]
    removed = row[('Removed', 'Ticker')]

    # remove the added companies
    if pd.notna(added):
        Tickers.discard(added)

    # bring the removed companies back
    if pd.notna(removed):
        Tickers.add(removed)

Tickers = pd.Series(sorted(Tickers))

start = '2020-01-01'
end = '2026-01-01'

prices , symbols = [] , []

for symbol in Tickers:
    df = yf.download(symbol,start=start, end=end)['Close']
    if not df.empty: 
        prices.append(df)
        symbols.append(symbol)

all_prices = pd.concat(prices, axis=1)
all_prices.columns = symbols

# remove stocks with missing prices
all_prices = all_prices.dropna(axis=1)
monthly_prices = all_prices.resample("ME").last()
monthly_return = monthly_prices.pct_change().dropna()

def mom(all_mtl_ret, lookback):

   # past cumulative return over the lookback period with a 1-month skip
    momentum = ((all_mtl_ret + 1).shift(1).rolling(lookback)
                .apply(lambda x: x.prod()-1))

    momentum.dropna(inplace=True)
    portfolio_returns = []
    dates = []

    for date, momentum_scores in momentum.iterrows():

        momentum_scores = momentum_scores.dropna()

        # top and bottom 10%
        n_stocks = int(len(momentum_scores)*0.10)
        winners = momentum_scores.nlargest(n_stocks).index
        losers = momentum_scores.nsmallest(n_stocks).index

        # hold after skip  1 month
        next_month = date + MonthEnd(2)

        # skip calculation after last month
        if next_month not in all_mtl_ret.index:
            continue

        winner_return = all_mtl_ret.loc[next_month, winners].mean()
        loser_return = all_mtl_ret.loc[next_month, losers].mean()

        # long winners and short losers
        portfolio_returns.append(winner_return - loser_return)
        dates.append(next_month)

    return pd.Series(portfolio_returns,index=dates)

# 6/1/1 strategy
momentum_returns = mom(monthly_return, 6)

# download Kenneth French risk-free rate
ff = web.DataReader(
    "F-F_Research_Data_Factors","famafrench",start=start,end=end)[0]

ff = ff / 100
ff.index = ff.index.to_timestamp("M")
rf = ff["RF"].reindex(momentum_returns.index).dropna()

# S&P 500 benchmark
S_P = yf.download("^GSPC",start=start,end=end,auto_adjust=True,progress=False)

SP500_monthly_returns = (S_P["Close"].squeeze().resample("ME").last()
                         .pct_change().dropna())

# match all dates
common_dates = (momentum_returns.index
                .intersection(SP500_monthly_returns.index)
                .intersection(rf.index))

momentum_returns = momentum_returns.loc[common_dates]
SP500_monthly_returns = SP500_monthly_returns.loc[common_dates]
rf = rf.loc[common_dates]

# cumulative returns
momentum_cumulative = (1 + momentum_returns).cumprod()
SP500_cumulative = (1 + SP500_monthly_returns).cumprod()
momentum_excess = momentum_returns - rf
SP500_excess = SP500_monthly_returns - rf

# performance metrics
momentum_total_return = float(momentum_cumulative.iloc[-1] - 1)
SP500_total_return = float(SP500_cumulative.iloc[-1] - 1)
momentum_volatility = momentum_returns.std() * np.sqrt(12)
SP500_volatility = SP500_monthly_returns.std() * np.sqrt(12)

momentum_sharpe = (momentum_excess.mean() * 12)/ momentum_volatility
SP500_sharpe = (SP500_excess.mean() * 12) / SP500_volatility
annual_returns = (1 + momentum_returns).groupby(momentum_returns.index.year).prod() - 1
annual_volatility = momentum_returns.groupby(momentum_returns.index.year).std() * np.sqrt(12)

# plot
plt.figure(figsize=(12,6))
plt.plot(momentum_cumulative.index, momentum_cumulative, label="Momentum Strategy")
plt.margins(x=0)  
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Momentum Strategy")
plt.legend()
plt.show()

performance_table = pd.DataFrame({
    "Annual Return": annual_returns,
    "Annual Volatility": annual_volatility})

print(performance_table)

comparison_table = pd.DataFrame(
    {"Momentum Strategy": [momentum_total_return,momentum_volatility,
            momentum_sharpe],
        "S&P 500": [SP500_total_return,SP500_volatility,SP500_sharpe]},
    index=["Total Return","Volatility","Sharpe Ratio"])

print("\nMomentum Strategy vs S&P 500 Performance\n")
print(comparison_table)