import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt
import indicators as indic
import marketsimcode as mktsim

def bb_indicator(symbol, sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 19
    df = get_data(symbol, dates)  # automatically adds SPY
    price_JPM = df[symbol]
    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm_JPM = indic.get_rolling_mean(price_JPM, window=window_size)
    # 2. Compute rolling standard deviation
    rstd_JPM = indic.get_rolling_std(price_JPM, window=window_size)

    # B-band index = (stock price - rolling_mean)/(2*std)
    bb_index = (price_JPM - rm_JPM )/ ( 1.2 *rstd_JPM)
    bb_index.fillna(0, inplace = True)

    trades_df = price_JPM.copy()
    trades_df[:] = 0
    # Strategy based on bb_index, fill in trades_df
    trades_df[bb_index > 1] = -1000
    trades_df[bb_index < -1] = +1000
    #trades_df.plot(title = "trades", label="trades" )
    #trades_df = trades_df.to_frame()
    #print trades_df # will be passed into marketSim
    return trades_df

# simple moving average
def sma_indicator(symbol, sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 21
    df = get_data(symbol, dates)  # automatically adds SPY
    price_JPM = df[symbol]
    rm_JPM = indic.get_rolling_mean(price_JPM, window=window_size)

    # sma index = (stock price - rolling_mean)/rolling_mean
    sma_index = (price_JPM - rm_JPM )/ rm_JPM
    sma_index.fillna(0, inplace = True)
    #print sma_index

    trades_df = price_JPM.copy()
    trades_df[:] = 0
    # Strategy based on sma_index, fill in trades_df

    trades_df[sma_index > 0.10] = -1000
    trades_df[sma_index < -0.10] = +1000
    #trades_df.plot(title = "trades", label="trades" )
    #trades_df = trades_df.to_frame()
    #print trades_df # will be passed into marketSim
    return trades_df

def test_policy_bb(symbol, sd, ed, start_value):
    trades_df = bb_indicator(symbol, sd, ed)
    commission = 9.95
    impact = 0.005
    # pass df_trade to marketsimcode to plot
    portvals = mktsim.compute_portvals(symbol, trades_df=trades_df, start_val= start_value, commission= commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "BollingerBandStrategy: Fund vs Benchmark")
    #indic.bollingerBand()

def test_policy_sma(symbol, sd, ed, start_value):
    trades_df = sma_indicator(symbol, sd, ed)
    commission = 9.95
    impact = 0.005
    portvals = mktsim.compute_portvals(symbol, trades_df=trades_df, start_val= start_value, commission= commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "SimpleMovingAverageStrategy: Fund vs Benchmark")

if __name__ == "__main__":
    start_value = 100000
    # in sample
    sd = dt.date(2008,1,1)
    ed = dt.date(2009,12,31)

    #out of sample
    #sd = dt.date(2010,1,1)
    #ed = dt.date(2011,12,31)
    symbol = ['JPM']

    test_policy_bb(symbol, sd, ed, start_value)
    test_policy_sma(symbol, sd, ed, start_value)


