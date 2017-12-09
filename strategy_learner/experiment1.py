"""
@Author: Yuanzheng Zhu (yzhu319)
experiment 1: compare Manual Strategy and ML Strategy
Commission =0 in all cases
Symbol = "JPM" for all cases
"""

import numpy as np
import math
import sys
import StrategyLearner as sl
import datetime as dt
import marketsimcode as mktsim
import indicators as indic
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt


def manual_bb_indicator(symbol, sd, ed):
    symbol = ['JPM']  # not used here... since only one stock
    dates = pd.date_range(sd, ed)
    window_size = 19
    df = get_data(['JPM'], dates)  # automatically adds SPY
    price_JPM = df['JPM']

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm_JPM = indic.get_rolling_mean(price_JPM, window=window_size)
    # 2. Compute rolling standard deviation
    rstd_JPM = indic.get_rolling_std(price_JPM, window=window_size)

    # B-band index = (stock price - rolling_mean)/(2*std)
    bb_index = (price_JPM - rm_JPM) / (1.2 * rstd_JPM)
    bb_index.fillna(0, inplace=True)

    trades_df = price_JPM.copy()
    trades_df[:] = 0
    # Strategy based on bb_index, fill in trades_df
    trades_df[bb_index > 1] = -1000
    trades_df[bb_index < -1] = +1000
    # trades_df.plot(title = "trades", label="trades" )
    trades_df = trades_df.to_frame()
    # print trades_df # will be passed into marketSim
    return trades_df


# simple moving average
def manual_sma_indicator(symbol, sd, ed):
    dates = pd.date_range(sd, ed)
    window_size = 21
    df = get_data(['JPM'], dates)  # automatically adds SPY
    price_JPM = df['JPM']
    rm_JPM = indic.get_rolling_mean(price_JPM, window=window_size)

    # sma index = (stock price - rolling_mean)/rolling_mean
    sma_index = (price_JPM - rm_JPM) / rm_JPM
    sma_index.fillna(0, inplace=True)
    # print sma_index

    trades_df = price_JPM.copy()
    trades_df[:] = 0
    # Strategy based on sma_index, fill in trades_df

    trades_df[sma_index > 0.10] = -1000
    trades_df[sma_index < -0.10] = +1000
    # trades_df.plot(title = "trades", label="trades" )
    trades_df = trades_df.to_frame()
    # print trades_df # will be passed into marketSim
    return trades_df


def manual_ema_indicator(symbol, sd, ed):
    dates = pd.date_range(sd, ed)
    window_size = 20  # use 10-period window
    df = get_data(['JPM'], dates)  # automatically adds SPY
    price_JPM = df['JPM']

    # init ema
    ema_JPM = price_JPM.copy()
    ema_JPM.fillna(0, inplace=True)
    # init the first value
    ema_JPM[window_size] = np.mean(price_JPM[0:window_size])
    # calc the weight factor
    multiplier = 2.0 / (window_size + 1)
    # get ema_index[i] from ema_index[i-1]
    for i in range(window_size + 1, len(price_JPM)):
        ema_JPM[i] = ema_JPM[i - 1] + multiplier * (price_JPM[i] - ema_JPM[i - 1])

    # sma index = (stock price - rolling_mean)/rolling_mean
    ema_index = (price_JPM - ema_JPM) / ema_JPM
    ema_index.fillna(0, inplace=True)

    trades_df = price_JPM.copy()
    trades_df[:] = 0

    trades_df[ema_index > 0.08] = -1000
    trades_df[ema_index < -0.08] = +1000
    # trades_df.plot(title = "trades", label="trades" )
    trades_df = trades_df.to_frame()
    # print trades_df # will be passed into marketSim
    return trades_df


def manual_test_policy_bb(symbol, sd, ed, start_value):
    trades_df = manual_bb_indicator(symbol, sd, ed)
    commission = 0
    impact = 0.005
    # pass df_trade to marketsimcode to plot
    portvals = mktsim.compute_portvals(trades_df=trades_df, start_val=start_value, commission=commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "BollingerBandStrategy: Fund vs Benchmark", symbol)


def manual_test_policy_sma(symbol, sd, ed, start_value):
    trades_df = manual_sma_indicator(symbol, sd, ed)
    commission = 0
    impact = 0.005
    portvals = mktsim.compute_portvals(trades_df=trades_df, start_val=start_value, commission=commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "SimpleMovingAverageStrategy: Fund vs Benchmark", symbol)


def manual_test_policy_ema(symbol, sd, ed, start_value):
    trades_df = manual_ema_indicator(symbol, sd, ed)
    commission = 0
    impact = 0.005
    portvals = mktsim.compute_portvals(trades_df=trades_df, start_val=start_value, commission=commission, impact=impact)
    mktsim.gen_plot(trades_df, portvals, "ExponentialMovingAverageStrategy: Fund vs Benchmark", symbol)


if __name__ == "__main__":
    start_value = 100000
    commission = 0
    impact = 0.005
    sym = "JPM"  # USE JPM for reports

    sd_insample = dt.date(2008, 1, 1)
    ed_insample = dt.date(2009, 12, 31)

    # train with in-sample data
    sl_learner = sl.StrategyLearner(verbose=True, impact=impact)
    sl_learner.addEvidence(symbol=sym, sd=sd_insample, ed=ed_insample, sv=10000)

    # test with in-sample data
    print "... Test In-sample with ML strategy ...\n"
    df_delta_trades_ML = sl_learner.testPolicy(symbol=sym, sd=sd_insample, ed=ed_insample, sv=10000)
    #convert df_delta_trades to df_trades
    df_trades_ML = df_delta_trades_ML.copy()
    for i in range(0, len(df_trades_ML)):
        if i == 0:
            df_trades_ML.iloc[i] = df_delta_trades_ML.iloc[i]
        else:
            df_trades_ML.iloc[i] = df_delta_trades_ML.iloc[i] + df_trades_ML.iloc[i - 1]

    portvals = mktsim.compute_portvals(trades_df=df_trades_ML, start_val=start_value,
                                                   commission=commission, impact=impact, symbol=sym)
    # pass df_trade to marketsimcode to plot
    mktsim.gen_plot(df_trades_ML, portvals, "Strategy Learner: Fund vs Benchmark", symbol=sym)

    print "... Test In-sample with manual strategy ...\n" # 3 plots
    manual_test_policy_bb(sym, sd_insample, ed_insample, start_value)
    manual_test_policy_sma(sym, sd_insample, ed_insample, start_value)
    manual_test_policy_ema(sym, sd_insample, ed_insample, start_value)

