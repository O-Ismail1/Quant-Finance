import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import matplotlib.pyplot as plt  
import requests
from io import StringIO

tickers = (pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        storage_options={"User-Agent": "Mozilla/5.0"})[0]["Symbol"]
    .str.replace(".", "-", regex=False).tolist())

df = yf.download(tickers,start="2020-01-01",end="2026-05-01",auto_adjust=True)["Close"]

# split data
train = df.loc["2020":"2024"]
test = df.loc["2025":"2026"]

# filter pairs using correlation
combi_df = pd.DataFrame(combinations(train.columns,2))
combi_df.columns = ['Stock1', 'Stock2']

combi_df['correlation'] = combi_df.apply(lambda row: np.corrcoef(train[row['Stock1']],
                                                                 train[row['Stock2']])[0,1],axis=1)

# keep only highly correlated pairs
combi_df = combi_df[combi_df.correlation > 0.98]

#Engle–Granger Cointegration Test
def do_reg(stock1,stock2):
    x = train[stock1].values
    y = train[stock2].values
    X = sm.add_constant(x)

    model = sm.OLS(y, X).fit()
    alpha, beta = model.params

    # regression residuals
    residuals = model.resid
    adf_stat, p_value = adfuller(residuals)[:2]

    return adf_stat, p_value, beta, alpha

# apply Engle-Granger test to all candidate pairs
combi_df[['adf_stat','p_value','beta','alpha']] = combi_df.apply(
    lambda row:pd.Series(do_reg(row['Stock1'],row['Stock2'])),axis=1)

cointegrated_pairs = combi_df[combi_df.p_value < 0.01].sort_values(by='adf_stat') 

pair = cointegrated_pairs.iloc[0]
stock1, stock2 =pair["Stock1"], pair["Stock2"]

#estimate rolling hedge ratio
rolling_window = 30

rolling_alpha = [np.nan] * rolling_window
rolling_beta = [np.nan] * rolling_window

for i in range(rolling_window, len(test)):
    past_data = test.iloc[i-rolling_window:i]

    x = past_data[stock1].values
    y = past_data[stock2].values
    X = sm.add_constant(x)

    model = sm.OLS(y,X).fit()
    rolling_alpha.append(model.params[0])
    rolling_beta.append(model.params[1])

rolling_params = pd.DataFrame({"Rolling Alpha": rolling_alpha,
                               "Rolling Beta": rolling_beta}, 
                               index=test.index)

# calculate spread z-score
spread = test[stock2] - (rolling_params["Rolling Alpha"] + rolling_params["Rolling Beta"] * test[stock1])

rolling_mean_spread = spread.rolling(30).mean()
rolling_std_spread = spread.rolling(30).std()
z_score = (spread - rolling_mean_spread) / rolling_std_spread

spread_data = pd.DataFrame({"Spread":spread,"Rolling Mean Spread": rolling_mean_spread,
                            "Rolling Std Spread": rolling_std_spread,"Z-Score": z_score}, 
                            index=test.index)

entry_threshold = 1.5
exit_threshold = 0.5

long_signal = spread_data["Z-Score"] <- entry_threshold
short_signal = spread_data["Z-Score"] > entry_threshold
exit_signal = spread_data["Z-Score"].abs() < exit_threshold

signals = pd.DataFrame({"Long Signal": long_signal,"Short Signal": short_signal,
                        "Exit Signal": exit_signal},index=test.index)

# backtest trading strategy
capital = 1000
position = 0
trade_log = []

#use next day open price

for i in range(1, len(test)-1):
    entry_bar = i+1

    if signals["Long Signal"].iloc[i] and position == 0:
        position = 1
        entry_price_stock1 = test[stock1].iloc[entry_bar]
        entry_price_stock2 = test[stock2].iloc[entry_bar]
        entry_date = test.index[entry_bar]

    elif signals["Short Signal"].iloc[i] and position == 0:
        position = -1
        entry_price_stock1 = test[stock1].iloc[entry_bar]
        entry_price_stock2 = test[stock2].iloc[entry_bar]
        entry_date = test.index[entry_bar]

    elif signals["Exit Signal"].iloc[i] and position != 0:
            exit_bar = i + 1
            exit_price_stock1 = test[stock1].iloc[exit_bar]
            exit_price_stock2 = test[stock2].iloc[exit_bar]
            trade_type = "Long" if position == 1 else "Short"

            beta_at_entry = rolling_params["Rolling Beta"].loc[entry_date]
            shares_stock2 = capital / entry_price_stock2
            shares_stock1 = (capital* beta_at_entry) / entry_price_stock1

            if position == 1:
                pnl = (shares_stock2 * (exit_price_stock2 - entry_price_stock2)) - \
               (shares_stock1 * (exit_price_stock1 - entry_price_stock1))

            else: 
                pnl = (shares_stock1 * (exit_price_stock1 - entry_price_stock1)) - \
                (shares_stock2 * (exit_price_stock2 - entry_price_stock2))

            trade_log.append({
                "Entry Date": entry_date,
                "Exit Date": test.index[exit_bar],
                "Trade Type": trade_type,
                "Entry Price Stock 1": entry_price_stock1,
                "Entry Price Stock 2": entry_price_stock2,
                "Exit Price Stock 1": exit_price_stock1,
                "Exit Price Stock 2": exit_price_stock2,
                "Profit/loss":pnl
    })
            position = 0

if position != 0:
    print(f"WARNING: Unclose {trade_type} position at end if data - excluded from results.")

trade_history = pd.DataFrame(trade_log)


# performance Metrics

initial_capital = 1000

# cumulative P&L
trade_history["Cumulative P&L"] = trade_history["Profit/loss"].cumsum()
trade_history["Equity"] = initial_capital + trade_history["Cumulative P&L"]
total_return = (trade_history["Equity"].iloc[-1] / initial_capital) - 1
number_of_trades = len(trade_history)
win_rate = (trade_history["Profit/loss"] > 0).mean()

# Average profit per trade
average_profit = trade_history["Profit/loss"].mean()
best_trade = trade_history["Profit/loss"].max()
worst_trade = trade_history["Profit/loss"].min()

equity_returns = trade_history["Equity"].pct_change().dropna()
annual_volatility = equity_returns.std() * np.sqrt(252)
sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252)

# Maximum drawdown
running_max = trade_history["Equity"].cummax()
drawdown = (trade_history["Equity"] - running_max) / running_max
max_drawdown = drawdown.min()


# Display results
performance = pd.DataFrame({
    "Metric": ["Initial Capital","Final Equity","Total Return","Number of Trades",
               "Win Rate","Average Trade P&L","Best Trade","Worst Trade","Annual Volatility",
               "Sharpe Ratio","Maximum Drawdown"],
    
    "Value": [initial_capital,trade_history["Equity"].iloc[-1],total_return,number_of_trades,
              win_rate,average_profit,best_trade,worst_trade,annual_volatility,
              sharpe_ratio,max_drawdown]})

print(performance.to_string(index=False))

# plot
plt.figure(figsize=(12,6))
plt.plot(
    trade_history["Exit Date"],
    trade_history["Profit/loss"].cumsum(),
    label="Cumulative P&L")

plt.scatter(trade_history["Entry Date"].iloc[0],
    trade_history["Profit/loss"].cumsum().iloc[0],
    color="red",s=80)
plt.xlabel("Date")
plt.ylabel("Cumulative P&L")
plt.title("Pairs Trading Cumulative P&L (2025-2026)")
plt.legend()

plt.xlim(trade_history["Exit Date"].min(),
    trade_history["Exit Date"].max())
plt.show()