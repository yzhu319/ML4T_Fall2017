"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    orders_df = pd.read_csv(orders_file, index_col = 'Date', parse_dates= True,na_values=['nan'])
    orders_df.sort_index(inplace=True)
    #print orders_df

    trade_dates = orders_df.index
    trade_dates = list(set(trade_dates))

    start_date = min(trade_dates)
    end_date = max(trade_dates)
    dates = pd.date_range(start_date, end_date)


    symbols = orders_df['Symbol'].values #list of stocks traded
    symbols = list(set(symbols))

    #print symbols

    prices_all = get_data(symbols, dates)  # automatically adds SPY
    # Fill NAN data
    prices_all.fillna(method = "ffill",inplace=True)
    prices_all.fillna(method = "bfill",inplace=True)
    prices = prices_all[symbols]  # only portfolio symbols
    #print prices

    _cash = pd.DataFrame(data= np.ones(len(dates)), index= dates, columns= ['_Cash'])
    prices_df = prices.join(_cash)
    #print prices_df

    trade_df = prices_df.copy() #mutable object!!
    trade_df[:] = 0.0

    #print orders_df

    for my_date, row in orders_df.iterrows():
        my_share = row['Shares']
        my_sym = row['Symbol']

        print orders_df
        print my_date
        print  my_sym

        if row['Order'] == 'BUY':
            my_share = my_share*1
        if row['Order'] == 'SELL':
            my_share = my_share*(-1)
        #fill in trade_df
        trade_df.loc[my_date, my_sym] += my_share
        unit_price = prices_df.get_value(my_date,my_sym)
        trade_df.loc[my_date, '_Cash'] += -my_share* unit_price
        # commission fees
        trade_df.loc[my_date, '_Cash'] -= (commission + abs(my_share)* unit_price*impact)

    #print trade_df

    holdings_df = trade_df.copy()
    # initialize
    holdings_df[:] = 0
    holdings_df.set_value(start_date, '_Cash', start_val)
    # update the 1st row
    holdings_df.iloc[0, :] += trade_df.iloc[0, :]
    # update the rest of rows in holdings_df
    for i in range(1,len(holdings_df.index)):
        holdings_df.iloc[i, :] += (holdings_df.iloc[i-1, :] + trade_df.iloc[i, :])
    #print holdings_df

    values_df = pd.DataFrame(holdings_df.values * prices_df.values, columns=prices_df.columns, index=prices_df.index)
    portvals = values_df.sum(axis = 1)
    #print values_df
    portvals_df = portvals.to_frame()

    return portvals_df

def author():
    return 'yzhu319'

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders_reddit.csv"
    sv = 100000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv,commission=0, impact=0)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
        print of
        print portvals.values
    else:
        "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(['IBM'], dates)  # automatically adds SPY
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    print prices_SPY

    # Get benchmark values
    SPY_daily_ret = prices_SPY/ prices_SPY.shift(1) - 1.0
    SPY_daily_ret = SPY_daily_ret[1:]

    print SPY_daily_ret

    SPY_ev = prices_SPY.iloc[-1]
    SPY_cr = (prices_SPY.iloc[-1] - prices_SPY.iloc[0]) / prices_SPY.iloc[0]

    SPY_adr = (SPY_daily_ret - 0).mean()
    SPY_sddr = (SPY_daily_ret * 1.0).std()
    SPY_sr = np.sqrt(252.0) * SPY_adr / SPY_sddr  # Sharpe Ratio, risk-adjusted return /std of daily return

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # ev, cr, adr, sddr, sr

    portvals_list = portvals.values.tolist()
    portvals_pdSeries = pd.Series(portvals_list)

    print type(portvals_pdSeries)

    port_daily_ret = portvals_pdSeries/ portvals_pdSeries.shift(1) - 1.0
    port_daily_ret = port_daily_ret[1:]

    ev = portvals_pdSeries.iloc[-1]
    cr = (portvals_pdSeries.iloc[-1] - portvals_pdSeries.iloc[0]) / portvals_pdSeries.iloc[0]

    print ev, cr

    adr = (port_daily_ret - 0).mean()
    sddr = (port_daily_ret * 1.0).std()

    sr = np.sqrt(252.0) * adr / sddr  # Sharpe Ratio, risk-adjusted return /std of daily return

#    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
#    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sr)
    print "Sharpe Ratio of SPY : {}".format(SPY_sr)
    print
    print "Cumulative Return of Fund: {}".format(cr)
    print "Cumulative Return of SPY : {}".format(SPY_cr)
    print
    print "Standard Deviation of Fund: {}".format(sddr)
    print "Standard Deviation of SPY : {}".format(SPY_sddr)
    print
    print "Average Daily Return of Fund: {}".format(adr)
    print "Average Daily Return of SPY : {}".format(SPY_adr)
    print
    print "Final Portfolio Value: {}".format(ev)

if __name__ == "__main__":
    test_code()
