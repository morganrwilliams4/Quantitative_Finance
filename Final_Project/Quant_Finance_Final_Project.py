#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:06:09 2024

@author: morganrhyswilliams
"""

# Case Study 1
## 1.1

#!pip install yfinance

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define tickers and time period
tickers = ["^SPX", "^IXIC", "^VIX", "^IRX"]
start_date = "2004-12-01"
end_date = "2024-11-30"

# Download data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date, interval="1d")

# Extract the 'Close' prices for S&P 500 Index and other variables
spx = data['Close']['^SPX']
ixic = data['Close']['^IXIC']
vix = data['Close']['^VIX']
irx = data['Close']['^IRX']

# Resample to get month-end data (last trading day of each month)
spx = spx.resample('M').last()
ixic = ixic.resample('M').last()
vix = vix.resample('M').last()
irx = irx.resample('M').last()

# Monthly Return Calculation for S&P 500 Index
monthly_return = spx.pct_change().dropna() * 100

# Display the first 5 and last 5 records of monthly returns
print("First 5 records of monthly returns:")
print(monthly_return.head())
print("Last 5 records of monthly returns:")
print(monthly_return.tail())

# 20-Year Compounded Return for S&P 500 Index
# The formula for compounded return is (final value / initial value) - 1
initial_value = spx.iloc[0]
final_value = spx.iloc[-1]
compounded_return = ((final_value / initial_value) - 1) * 100
print("20-Year Compounded Return for S&P 500 Index:", compounded_return)

# Annualized Return (Natural Logarithm method)
# Formula: log(final_value / initial_value) / number_of_years
years = len(spx) / 12  # Convert the total number of months into years
annualized_return = (np.log(final_value / initial_value) / years) * 100
print("Annualized Return for S&P 500 Index:", annualized_return)

##1.2

# Moving Averages for NASDAQ Composite ("^IXIC")

# Calculate 3-month and 36-month moving averages
ixic_ma3 = ixic.rolling(window=3).mean()
ixic_ma36 = ixic.rolling(window=36).mean()

# Plot NASDAQ Composite price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(ixic, label="NASDAQ Composite", color="blue")
plt.plot(ixic_ma3, label="3-Month Moving Average", color="orange")
plt.plot(ixic_ma36, label="36-Month Moving Average", color="green")
plt.title("NASDAQ Composite with 3-Month and 36-Month Moving Averages")
plt.xlabel("Date")
plt.ylabel("NASDAQ Composite Price(USD)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#Case Study 2
##2.1

# Option Spread Analysis
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Parameters
S = np.arange(50, 155, 5)  # Stock price range
K1, K2, K3 = 90, 100, 110  # Strike prices
T = 1  # Time to maturity in years
r = 0.05  # Risk-free rate
sigma = 0.25  # Volatility
S0 = 100  # Current stock price

# Calculate option premiums
call_K1 = black_scholes_call(S0, K1, T, r, sigma)
call_K2 = black_scholes_call(S0, K2, T, r, sigma)
put_K2 = black_scholes_put(S0, K2, T, r, sigma)
put_K3 = black_scholes_put(S0, K3, T, r, sigma)

# Strategy payoffs at expiration
# Bull Call Spread
payoff_bull_call = np.maximum(S - K1, 0) - np.maximum(S - K2, 0)
profit_bull_call = payoff_bull_call - (call_K1 - call_K2)

# Bear Put Spread
payoff_bear_put = np.maximum(K3 - S, 0) - np.maximum(K2 - S, 0)
profit_bear_put = payoff_bear_put - (put_K3 - put_K2)

# Straddle
payoff_straddle = np.maximum(S - K2, 0) + np.maximum(K2 - S, 0)
profit_straddle = payoff_straddle - (call_K2 + put_K2)

# Calculate breakeven points and max profit/loss
# Bull Call Spread
breakeven_bull_call = K1 + (call_K1 - call_K2)
max_profit_bull_call = K2 - K1 - (call_K1 - call_K2)
max_loss_bull_call = -(call_K1 - call_K2)

# Bear Put Spread
breakeven_bear_put = K3 - (put_K3 - put_K2)
max_profit_bear_put = K3 - K2 - (put_K3 - put_K2)
max_loss_bear_put = -(put_K3 - put_K2)

# Straddle
breakeven_straddle_low = K2 - (call_K2 + put_K2)
breakeven_straddle_high = K2 + (call_K2 + put_K2)
max_profit_straddle = 50 - (call_K2 + put_K2)
max_loss_straddle = -(call_K2 + put_K2)

# Plotting
# Bull Call Spread
plt.figure()
plt.plot(S, payoff_bull_call, label="Payoff", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Bull Call Spread - Payoff")
plt.xlabel("Stock Price at Expiration ($S$)")
plt.ylabel("Payoff")
plt.grid()
plt.legend()

plt.figure()
plt.plot(S, profit_bull_call, label="Profit/Loss", color="green")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(breakeven_bull_call, color="red", linestyle="--", label="Breakeven")
plt.title("Bull Call Spread - Profit/Loss")
plt.xlabel("Stock Price at Expiration ($S$)")
plt.ylabel("Profit/Loss")
plt.grid()
plt.legend()

# Bear Put Spread
plt.figure()
plt.plot(S, payoff_bear_put, label="Payoff", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Bear Put Spread - Payoff")
plt.xlabel("Stock Price at Expiration ($S$)")
plt.ylabel("Payoff")
plt.grid()
plt.legend()

plt.figure()
plt.plot(S, profit_bear_put, label="Profit/Loss", color="green")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(breakeven_bear_put, color="red", linestyle="--", label="Breakeven")
plt.title("Bear Put Spread - Profit/Loss")
plt.xlabel("Stock Price at Expiration ($S$)")
plt.ylabel("Profit/Loss")
plt.grid()
plt.legend()

# Straddle
plt.figure()
plt.plot(S, payoff_straddle, label="Payoff", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Straddle - Payoff")
plt.xlabel("Stock Price at Expiration ($S$)")
plt.ylabel("Payoff")
plt.grid()
plt.legend()

plt.figure()
plt.plot(S, profit_straddle, label="Profit/Loss", color="green")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(breakeven_straddle_low, color="red", linestyle="--", label="Breakeven Low")
plt.axvline(breakeven_straddle_high, color="red", linestyle="--", label="Breakeven High")
plt.title("Straddle - Profit/Loss")
plt.xlabel("Stock Price at Expiration ($S$)")
plt.ylabel("Profit/Loss")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Print breakeven points and max profit/loss
print("Bull Call Spread:")
print(f"  Breakeven: {breakeven_bull_call:.2f}")
print(f"  Max Profit: {max_profit_bull_call:.2f}")
print(f"  Max Loss: {max_loss_bull_call:.2f}\n")

print("Bear Put Spread:")
print(f"  Breakeven: {breakeven_bear_put:.2f}")
print(f"  Max Profit: {max_profit_bear_put:.2f}")
print(f"  Max Loss: {max_loss_bear_put:.2f}\n")

print("Straddle:")
print(f"  Breakeven Low: {breakeven_straddle_low:.2f}")
print(f"  Breakeven High: {breakeven_straddle_high:.2f}")
print(f"  Max Profit: {max_profit_straddle:.2f}")
print(f"  Max Loss: {max_loss_straddle:.2f}")
