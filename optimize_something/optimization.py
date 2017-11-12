"""MC1-P2: Optimize a portfolio."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
import datetime as dt
from util import get_data, plot_data

def calc_stats(allocs, prices_norm):
    """once get an allocation, calc stats of the portfolio"""
    prices_alloc = prices_norm * allocs
    pos_alloc = 1.0 * prices_alloc
    port_val = pos_alloc.sum(axis=1) # row sum, axis = 1; column sum, axis = 0
    
    # Get portfolio statistics (note: std_daily_ret = volatility)
    ev = port_val[-1]
    cr = (port_val[-1] - port_val[0])/port_val[0]
    
    port_daily_ret = 1.0* port_val/port_val.shift(1) -1.0
    port_daily_ret = port_daily_ret[1:]
    
    adr = (port_daily_ret - 0).mean() 
    sddr = (port_daily_ret * 1.0).std()
    sr = np.sqrt(252.0) * adr / sddr # Sharpe Ratio, risk-adjusted return /std of daily return 
    return cr, adr, sddr, sr, port_val

def calc_std(allocs, prices_norm):
    """Given a vector allocs, return the std of the portfolio"""
    prices_alloc = prices_norm * allocs
    pos_alloc = 1.0 * prices_alloc
    port_val = pos_alloc.sum(axis=1) # row sum, axis = 1; column sum, axis = 0
        
    port_daily_ret = 1.0* port_val/port_val.shift(1) -1.0
    port_daily_ret = port_daily_ret[1:]
    adr = (port_daily_ret - 0).mean() 
    sddr = (port_daily_ret * 1.0).std()     #minimize sddr#
    
    return sddr


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1),     syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Fill NAN data
    prices_all.fillna(method = "ffill",inplace=True)
    prices_all.fillna(method = "bfill",inplace=True)

    ###Add codes...###
    # find the allocations for the optimal portfolio    
    
    # initial guess: same allocation
    num_stocks = len(syms)
    allocs0 = np.ones(num_stocks) * (1.0/num_stocks)
    
    # Get daily portfolio value
    port_val = prices_SPY 
    prices_norm = prices / prices.iloc[0]
    
    # calc the allocs for minimized std
    
    # bounds for minimization, for every stock, allocation [0,1]
    bnds = tuple([(0,1)]*num_stocks)
    # constrains, sum(allocs)==1
    cons = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    minimize_results = spo.minimize(calc_std, allocs0, args=(prices_norm), method='SLSQP', bounds = bnds, constraints = cons)
    allocs = minimize_results.x
    #print(allocs)
    #print(allocs.sum())
    
    [cr, adr, sddr, sr, port_val] = calc_stats(allocs, prices_norm)
    
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:              
        prices_SPY_norm = prices_SPY / prices_SPY[0]
        port_val_norm = port_val / port_val[0]
        fig = plt.figure()
        
        #retain matplotlib axis object ax
        ax = prices_SPY_norm.plot(title = "Daily portfolio value and SPY",label="SPY")
        port_val_norm.plot(label="Portfolio",ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="upper right")
        #plt.show()
        fig.savefig('report.pdf')
        pass

    return allocs, cr, adr, sddr, sr


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    #symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']
    symbols = ['IBM','X','GLD']
    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,        syms = symbols,         gen_plot = True)

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
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()

