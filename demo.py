import streamlit as st

import pandas as pd
import numpy as np

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.black_litterman import BlackLittermanModel

import warnings
warnings.filterwarnings("ignore")

# st.write("""Stock code list""")

# efficient frontier
def get_efficient_frontier(d, method="sample"):
    stock_prices = pd.DataFrame(columns=list(d.keys()))
    for ts_code, df in d.items():
        stock_prices[ts_code] = df["close"]
    stock_prices = stock_prices.iloc[::-1]
    if method == "sample":
        mu = expected_returns.mean_historical_return(stock_prices) # Calculate annualised mean (daily) historical return from input (daily) asset prices.
        cov_matrix = risk_models.risk_matrix(stock_prices, method="sample_cov") # Calculate the annualised sample covariance matrix of (daily) asset returns.
    elif method == "time_exp":
        mu = expected_returns.ema_historical_return(stock_prices) # Calculate the exponentially-weighted mean of (daily) historical returns, giving higher weight to more recent data.
        cov_matrix = risk_models.risk_matrix(stock_prices, method="exp_cov") # Estimate the exponentially-weighted covariance matrix, which gives greater weight to more recent data.
    elif method == "CAPM":
        mu = expected_returns.capm_return(stock_prices) # Compute a return estimate using the Capital Asset Pricing Model.
        cov_matrix = risk_models.risk_matrix(stock_prices, method="sample_cov") # Estimate the semicovariance matrix, i.e the covariance given that the returns are less than the benchmark.
    ef = EfficientFrontier(mu, cov_matrix)
    return ef, mu, cov_matrix

def true_annual_return(weight_dict, filtered_stocks_val_dict): # weight data type: ordered dict
    weight = np.array(list(weight_dict.values())).reshape(-1,1)
    stocks = list(weight_dict.keys())
    daily_return_rate = []
    for ts_code in stocks:
        daily_return_rate.append(np.average(filtered_stocks_val_dict[ts_code]["pct_chg"]))
    daily_return_rate = np.array(daily_return_rate).reshape(-1,1)
    annual_return = np.dot(weight.T, daily_return_rate) 
    
    return annual_return[0][0] * 2.52 # 252 working days for a year

## the efficient frontier after updated by bl model.
def get_efficient_frontier_bl(predicted_views, filtered_stocks):
    filtered_ts_code = list(filtered_stocks.keys())
    Sigma = np.array(cov_matrix1)
    market_weights = np.array(list(ef1.clean_weights().values())).reshape(-1, 1)
    risk_aversion = ef1.portfolio_performance(risk_free_rate=rf)[2] / np.sqrt(np.dot(np.dot(market_weights.T, Sigma), market_weights))
    Pi = risk_aversion * np.dot(Sigma, market_weights)
    P = np.diag(np.diag(np.ones((len(filtered_ts_code), len(filtered_ts_code))))) # picking matrix
    Q = np.array(predicted_views).reshape(-1, 1) # view matrix (to be changed)
    tau = 1 / (252 * 10) # 1/T
    Omega = np.diag(np.diag(np.dot(np.dot(P, tau * Sigma), P.T))) # Omega = P*(tau*Sigma)*P'

    bl = BlackLittermanModel(cov_matrix=Sigma, pi=Pi, Q=Q, P=P, omega=Omega, tau=tau)
    returns = bl.bl_returns()
    returns.index = filtered_ts_code
    cov = bl.bl_cov()
    ef1_post = EfficientFrontier(returns, cov)
    post_weight_dict1 = ef1_post.max_sharpe(risk_free_rate=rf)

    return post_weight_dict1, ef1_post.portfolio_performance(verbose=True, risk_free_rate=rf)

# Get corresponding predicted views from the file.
def read_predicted_views(stock_codes):
    filtered_stocks = pd.read_pickle('filtered_stocks_daily.pickle')
    filtered_ts_code = list(filtered_stocks.keys())

    with open("predictions.txt", "r") as f:
        exp_rtn =  list(map(float, f.read().splitlines()))
    exp_rtn = [x * 252 for x in exp_rtn]
    
    stock_index = [filtered_ts_code.index(code) for code in stock_codes]
    
    return [exp_rtn[index] for index in stock_index]

    
        
def demo_options():
    options = st.multiselect(
    'Select stocks from the list for the portfolio',
    ['600029.SH',
    '600115.SH',
    '600132.SH',
    '600139.SH',
    '600242.SH',
    '600258.SH',
    '600303.SH',
    '600330.SH',
    '600399.SH',
    '600408.SH'],
    max_selections=5
    )

    st.write('You selected:', options)
             
    if st.button('Submit'):
        if len(options) == 5:
            return options
        else:
            st.write("Error: Please enter 5 stocks")

    return []


if __name__ == "__main__":
    stock_codes = demo_options()

    col1, col2 = st.columns(2)

    rf = 0.02   # Should be determined by input 
    freq = 252

    if len(stock_codes) > 0:
        filtered_stocks = pd.read_pickle("filtered_stocks_daily.pickle")
        filtered_stocks_val = pd.read_pickle("filtered_stocks_val.pickle")

        for key in list(filtered_stocks.keys()): 
            if key not in stock_codes:
                filtered_stocks.pop(key)

        ef1, mu1, cov_matrix1 = get_efficient_frontier(filtered_stocks, method="sample")
        prior_weight_dict1 = ef1.max_sharpe(risk_free_rate=rf)
        exp_rtn, ann_vol, sharpe_ratio = ef1.portfolio_performance(verbose=True, risk_free_rate=rf)

        with col1:
            st.markdown("""**Result without predicted views**""")
            for stock in prior_weight_dict1.keys():
                st.write(stock, ": ", prior_weight_dict1[stock])
            st.write("Expected Annual Return: ", exp_rtn)
            st.write("Annual Volatility: ", ann_vol)
            st.write("Sharpe Ratio: ", sharpe_ratio)
            st.write("Actual Annual Market Return", true_annual_return(prior_weight_dict1, filtered_stocks_val))
            
        predicted_views = read_predicted_views(filtered_stocks)
        post_weight_dict1, bl_performance = get_efficient_frontier_bl(predicted_views, filtered_stocks)
        exp_rtn_bl, ann_vol_bl, sharpe_ratio_bl = bl_performance

        with col2:
            st.markdown("""**Result updated by predicted views**""")
            for stock in post_weight_dict1.keys():
                st.write(stock, ": ", post_weight_dict1[stock])
            st.write("Expected Annual Return: ", exp_rtn_bl)
            st.write("Annual Volatility: ", ann_vol_bl)
            st.write("Sharpe Ratio: ", sharpe_ratio_bl)
            st.write("Actual Annual Market Return", true_annual_return(post_weight_dict1, filtered_stocks_val))




            