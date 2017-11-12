# Best possible Strategy, if we can see future:
# if JPM_price_tomorrow- JPM_price_today > 0, position = long 1000 JPM today
# if JPM_price_tomorrow- JPM_price_today < 0, position = short 1000 JPM today
import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data

import marketsimcode as mktsim

def testPolicy(symbol, start_date, end_date, start_value):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(['JPM'], dates)  # automatically adds SPY
    prices_JPM = prices_all['JPM']

    trades_df = prices_JPM.copy()
    trades_df[:] = 0

    #update from day 1 to day [N-1]
    for i in range(0, len(trades_df)-1):
        if prices_JPM.iloc[i+1] > prices_JPM.iloc[i]:
            trades_df.iloc[i] = +1000
        elif prices_JPM.iloc[i+1] < prices_JPM.iloc[i]:
            trades_df.iloc[i] = -1000
        else:
            trades_df.iloc[i] = 0
    #print trades_df
    #convert to df
    trades_df = trades_df.to_frame()
    return trades_df

if __name__ == "__main__":
    symbol = 'JPM'
    start_value = 100000
    sd = dt.date(2008, 1, 1)
    ed = dt.date(2009, 12, 31)
    commission = 0
    impact = 0
    trades_df = testPolicy(symbol= symbol, start_date= sd, end_date=ed, start_value= start_value)

    #print trades_df
    # pass df_trade to marketsimcode to plot
    portvals = mktsim.compute_portvals(trades_df=trades_df, start_val= start_value, commission= commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "BestPossibleStrategy: Fund vs Benchmark")
