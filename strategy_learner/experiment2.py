"""
@Author: Yuanzheng Zhu (yzhu319)
experiment 2: Study the impact parameter on ML strategy
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
import random

if __name__ == "__main__":
    # Set fixed seed for repetability
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    start_value = 100000
    commission = 0

    # change impact value
    impact = 0.010
    sym = "JPM"  # USE JPM for reports

    sd_insample = dt.date(2008, 1, 1)
    ed_insample = dt.date(2009, 12, 31)

    # train with in-sample data
    sl_learner = sl.StrategyLearner(verbose=True, impact=impact)
    sl_learner.addEvidence(symbol=sym, sd=sd_insample, ed=ed_insample, sv=10000)

    # test with in-sample data
    print "... Test In-sample with ML strategy ...\n"
    df_delta_trades_ML = sl_learner.testPolicy(symbol=sym, sd=sd_insample, ed=ed_insample, sv=10000)

    #calc number of trades, non-zero entries in df_delta_trades_ML
    tot_trades = (df_delta_trades_ML[sym] != 0).sum()
    print "total trades are {} for impact = {}".format(tot_trades, impact)

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


