#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:27:40 2024

@author: morganrhyswilliams
"""

import numpy as np
import matplotlib.pyplot as plt


#European options pricing function
def binomial_tree_eu(S, K, T, r, sigma, N=100, option_type='call'):
    """
    Calculate the European call option price using the binomial tree model.
    Parameters:
        - S: Current stock price
        - K: Option strike price
        - T: Time to expiration (in years)
        - r: Risk-free interest rate
        - sigma: Volatility of the underlying stock
        - n: Number of time steps in the binomial tree
    Returns:
        - European call option price
    """
    dt = T / N #Time step
    u = np.exp(sigma * np.sqrt(dt)) #Up factor
    d = 1 / u # Down factor
    p = (np.exp(r * dt) - d) / (u - d) #Probability of going up
    q = 1 - p
    #Build the binomial tree
    option_tree = [[0] * (i + 1) for i in range(N + 1)]
    
    #Calculate the payoff at maturity
    for j in range(N + 1):
        if option_type == 'call':
            option_tree[N][j] = max(0, S * (u ** (N - j)) * (d ** j) - K)
        elif option_type == 'put':
            option_tree[N][j] = max(0, K - S * (u ** (N - j)) * (d ** j))
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")
            
    #Backwards induction for European option
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_tree[i][j] = np.exp(-r * dt) * (p * option_tree[i + 1][j] + q * option_tree[i + 1][j + 1])
    return option_tree[0][0]

#American options pricing function
def binomial_tree_am(S0, K, T, r, sigma, N=100, option_type="call"):
    """
    Calculate the American option price using the binomial tree model.
    Parameters:
        - S0: Initial stock price
        - K: Option strike price
        - T: Time to expiration (in years)
        - r: Risk-free interest rate
        - sigma: Volatility of the underlying stock
        - N: Number of time steps in the binomial tree
        - option_type: Type of option ('call' or 'put')
    Returns:
        - Option price
    """
    dt = T / N  #Time step
    u = np.exp(sigma * np.sqrt(dt))  #Up factor
    d = 1 / u  #Down factor
    p = (np.exp(r * dt) - d) / (u - d)  #Probability of going up
    q = 1 - p  #Probability of going down
    #Build the binomial tree for option prices
    option_tree = [[0] * (i + 1) for i in range(N + 1)]

    #Calculate the payoff at maturity
    for j in range(N + 1):
        if option_type == "call":
            option_tree[N][j] = max(0, S0 * (u ** (N - j)) * (d ** j) - K)
        elif option_type == "put":
            option_tree[N][j] = max(0, K - S0 * (u ** (N - j)) * (d ** j))
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")

    #Backward induction for American option (consider early exercise)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            #Intrinsic value (for early exercise)
            if option_type == "call":
                intrinsic_value = max(S0 * (u ** (i - j)) * (d ** j) - K, 0)
            elif option_type == "put":
                intrinsic_value = max(K - S0 * (u ** (i - j)) * (d ** j), 0)

            #Continuation value (value if not exercised early)
            continuation_value = np.exp(-r * dt) * (p * option_tree[i + 1][j] + q * option_tree[i + 1][j + 1])

            #The option value at node (i, j) is the maximum of exercising early or continuing
            option_tree[i][j] = max(intrinsic_value, continuation_value)

    return option_tree[0][0]


#Parameters
S = np.arange(50, 155, 5)  #Stock prices range
K = 100  #Strike price
T = 1  #Time to maturity
r = 0.05  #Risk-free rate
sigma = 0.3  #Volatility of Xmazon
N = 100  #Number of time steps

#Calculate option prices
eu_call_prices = [binomial_tree_eu(s, K, T, r, sigma, N, "call") for s in S]
am_call_prices = [binomial_tree_am(s, K, T, r, sigma, N, "call") for s in S]
eu_put_prices = [binomial_tree_eu(s, K, T, r, sigma, N, "put") for s in S]
am_put_prices = [binomial_tree_am(s, K, T, r, sigma, N, "put") for s in S]

#Payoff functions
call_payoff = np.maximum(S - K, 0)
put_payoff = np.maximum(K - S, 0)

#Plotting
plt.figure(figsize=(14, 10))

#Figure 1: Call Options
plt.subplot(2, 1, 1)
plt.plot(S, call_payoff, label="Call Payoff", linestyle="--")
plt.plot(S, eu_call_prices, label="European Call Price", marker="o")
plt.plot(S, am_call_prices, label="American Call Price", marker="x")
plt.title("Call Option Payoff and Prices")
plt.xlabel("Stock Price (S)")
plt.ylabel("Price / Payoff")
plt.legend()
plt.grid()

#Figure 2: Put Options
plt.subplot(2, 1, 2)
plt.plot(S, put_payoff, label="Put Payoff", linestyle="--")
plt.plot(S, eu_put_prices, label="European Put Price", marker="o")
plt.plot(S, am_put_prices, label="American Put Price", marker="x")
plt.title("Put Option Payoff and Prices")
plt.xlabel("Stock Price (S)")
plt.ylabel("Price / Payoff")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
