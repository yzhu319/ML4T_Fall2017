"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

import matplotlib.pyplot as plt

# Function to compute and return daily return
def calc_daily_returns(df):
    daily_returns = 1.0* df/df.shift(1) -1.0
    return daily_returns
    
    '''
    num_days = len(port_val)-1
    port_daily_ret = np.ones(num_days) 
    for i in range(0,num_days):
        port_daily_ret[i] = port_val[i+1] / port_val[i] -1
    '''
    
    '''
    daily_returns = df.copy() #df of same size: N x 1
    daily_returns[1:] = df[1:]/df[:-1].values -1
    daily_returns.ix[0] = 0
    '''
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values

    '''
    prices_norm = prices / prices.iloc[0]
    port_price = 0 # Nx1 array, N is timestamps; as if a ETF
    for i in range(0,len(allocs)):
        port_price += allocs[i] * prices_norm.iloc[:,i]
        port_val = sv * port_price # portfolio value for each day
    '''
    
    prices_norm = prices / prices.iloc[0]
    prices_alloc = prices_norm * allocs
    pos_alloc = sv * prices_alloc
    port_val = pos_alloc.sum(axis=1) # row sum, axis = 1; column sum, axis = 0

    print port_val
    # Get portfolio statistics (note: std_daily_ret = volatility)
    #ev, cr, adr, sddr, sr
    ev = port_val[-1]
    cr = (port_val[-1] - port_val[0])/port_val[0]

    port_daily_ret = calc_daily_returns(port_val)
    port_daily_ret = port_daily_ret[1:]
    
    adr = np.average(port_daily_ret - rfr * sf/252.0) 
    sddr = np.std(port_daily_ret) # std of daily return
    sr = np.sqrt(252) * adr / sddr # Sharpe Ratio, risk-adjusted return /std of daily return

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
                
        prices_SPY_norm = prices_SPY / prices_SPY[0]
        port_val_norm = port_val / port_val[0]
        fig = plt.figure()
        #df_temp = pd.concat([port_val_norm, prices_SPY_norm], keys=['Portfolio', 'SPY'], axis=1)
        
        #retain matplotlib axis object ax
        ax = prices_SPY_norm.plot(title = "Daily portfolio and SPY",label="SPY")
        port_val_norm.plot(label="Portf",ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        #plt.show()
        fig.savefig('plot.png')
        pass

    
    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
