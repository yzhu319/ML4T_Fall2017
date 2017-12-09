"""
@Author: Yuanzheng Zhu (yzhu319)
Modified ManualStrategy code based on ManualStrategy project
Instead of setting criteria of a indicator (like bb_indicator>1) to trigger LONG or SHORT action,
We use ML method to determine
Input: symbol(as string like 'JPM'), sd, ed
Output: for each indicator, return a pd frame -- indicator value VS date

"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt
import indicators as indic
#import marketsimcode as mktsim

# New function, gather indicators' values and Y value (N-day return) converted to (-1,0,+1)
# Input, ticker, sd, ed
# Output, a pd frame to be used to train Learner, col: X1, X2, X3, Y; row: 2009-01-01, 2009-01-02,...
# X1, X2, X3 is the indicator values, or "features", combined output is equivalent to /assess_learners/DATA/Istanbul.csv

def testing_data(symbol, sd,ed):

    X_bb = bb_indicator(symbol, sd,ed)
    X_sma = sma_indicator(symbol, sd,ed)
    X_ema = ema_indicator(symbol, sd,ed)

    data4testX = (X_bb.join(X_sma)).join(X_ema)

    #print data4learn
    return data4testX

def training_data(symbol, sd,ed, impact):

    X_bb = bb_indicator(symbol, sd,ed)
    X_sma = sma_indicator(symbol, sd,ed)
    X_ema = ema_indicator(symbol, sd,ed)

    dates = pd.date_range(sd, ed)
    df = get_data([symbol], dates)  # automatically adds SPY
    price = df[symbol]

    Y_Nday_ret = price.copy()
    Y_Nday_ret[:] = 0

    #Parameters
    Nday = 8
    YBUY = 0.05
    YSELL = -0.05

    #print len(price)
    for i in range(0, len(price)-Nday):
        Y_Nday_ret[i] = price[i+Nday]/ price[i] - 1.0

    Y_Nday_ret = Y_Nday_ret.to_frame()

    Y = Y_Nday_ret.copy()
    Y[:] = 0
    #convert continuous Nday returns to +1, -1 or 0
    Y[Y_Nday_ret > (YBUY + impact)] = 1
    Y[Y_Nday_ret < (YSELL - impact)] = -1

    data4learn = ((X_bb.join(X_sma)).join(X_ema)).join(Y)

    #print data4learn
    return data4learn[21:-Nday]
    #return data4learn

def bb_indicator(symbol, sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 19
    df = get_data([symbol], dates)  # automatically adds SPY
    price = df[symbol]

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm = indic.get_rolling_mean(price, window=window_size)
    # 2. Compute rolling standard deviation
    rstd = indic.get_rolling_std(price, window=window_size)


    # B-band index = (stock price - rolling_mean)/(2*std)
    bb_index = (price - rm )/ ( 1.2 *rstd)
    bb_index.name = "X_bb"

    bb_index.fillna(0, inplace = True)

    bb_index = bb_index.to_frame()
    #print bb_index

    return bb_index

# simple moving average
def sma_indicator(symbol, sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 21
    df = get_data([symbol], dates)  # automatically adds SPY
    price = df[symbol]
    rm = indic.get_rolling_mean(price, window=window_size)

    # sma index = (stock price - rolling_mean)/rolling_mean
    sma_index = (price - rm )/ rm

    sma_index.name = "X_sma"
    sma_index.fillna(0, inplace = True)
    #print sma_index

    #trades_df = price.copy()
    #trades_df[:] = 0
    # Strategy based on sma_index, fill in trades_df

    #trades_df[sma_index > 0.10] = -1000
    #trades_df[sma_index < -0.10] = +1000
    #trades_df.plot(title = "trades", label="trades" )
    #trades_df = trades_df.to_frame()
    #print trades_df # will be passed into marketSim
    return sma_index

def ema_indicator(symbol, sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 20 #use 10-period window
    df = get_data([symbol], dates)  # automatically adds SPY
    price = df[symbol]

    # init ema
    ema = price.copy()
    ema.fillna(0, inplace = True)
    # init the first value
    ema[window_size] = np.mean(price[0:window_size])
    # calc the weight factor
    multiplier = 2.0/ (window_size+1)
    # get ema_index[i] from ema_index[i-1]
    for i in range(window_size+1,len(price)):
        ema[i] = ema[i-1] + multiplier *(price[i] - ema[i-1])

    # sma index = (stock price - rolling_mean)/rolling_mean
    ema_index = (price - ema )/ ema
    ema_index.name = "X_ema"
    ema_index.fillna(0, inplace = True)

    #trades_df = price.copy()
    #trades_df[:] = 0

    #trades_df[ema_index > 0.08] = -1000
    #trades_df[ema_index < -0.08] = +1000
    #trades_df.plot(title = "trades", label="trades" )
    #trades_df = trades_df.to_frame()
    #print trades_df # will be passed into marketSim
    return ema_index

def test_policy_bb(symbol, sd, ed, start_value):
    trades_df = bb_indicator(symbol, sd, ed)
    commission = 9.95
    impact = 0.005
    # pass df_trade to marketsimcode to plot
    portvals = mktsim.compute_portvals(trades_df=trades_df, start_val= start_value, commission= commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "BollingerBandStrategy: Fund vs Benchmark")
    #indic.bollingerBand()

def test_policy_sma(symbol, sd, ed, start_value):
    trades_df = sma_indicator(symbol, sd, ed)
    commission = 9.95
    impact = 0.005
    portvals = mktsim.compute_portvals(trades_df=trades_df, start_val= start_value, commission= commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "SimpleMovingAverageStrategy: Fund vs Benchmark")

def test_policy_ema(symbol, sd, ed, start_value):
    trades_df = ema_indicator(symbol, sd, ed)
    commission = 9.95
    impact = 0.005
    portvals = mktsim.compute_portvals(trades_df=trades_df, start_val= start_value, commission= commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "ExponentialMovingAverageStrategy: Fund vs Benchmark")

if __name__ == "__main__":
    start_value = 100000
    # in sample
    sd = dt.date(2008,1,1)
    ed = dt.date(2008,5,1)

    #out of sample
    #sd = dt.date(2010,1,1)
    #ed = dt.date(2011,12,31)
    symbol = 'JPM'

    X_bb = bb_indicator(symbol, sd, ed)
    X_sma = sma_indicator(symbol, sd, ed)
    X_ema = ema_indicator(symbol, sd, ed)
    Y = training_data(symbol,sd,ed)

    print X_bb
    print Y
    data4learn = ((X_bb.join(X_sma)).join(X_ema)).join(Y)

    print data4learn
    #test_policy_sma(symbol, sd, ed, start_value)
    #test_policy_ema(symbol, sd, ed, start_value)


