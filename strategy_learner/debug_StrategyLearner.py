"""
@Author: Yuanzheng Zhu (yzhu319)
Debug ML Strategy

"""

import numpy as np
import math
import sys
import StrategyLearner as sl
import datetime as dt
import marketsimcode as mktsim
import indicators as indic

if __name__ == "__main__":
    start_value = 100000
    commission = 0
    impact = 0.005
    sym = "JPM" #USE JPM for reports

    sd_insample = dt.date(2008,1,1)
    ed_insample = dt.date(2009,12,31)

    # train with in-sample data
    sl_learner = sl.StrategyLearner(verbose=True, impact=impact)
    sl_learner.addEvidence(symbol=sym, sd=sd_insample, ed=ed_insample, sv=10000)

    # test with in-sample data
    print "Test In-sample with ML strategy\n"
    df_delta_trades_ML = sl_learner.testPolicy(symbol=sym,sd=sd_insample, ed=ed_insample, sv=10000)
    # pass df_trade to marketsimcode to plot
    portvals = mktsim.compute_portvals_trade_delta(trades_delta_df=df_delta_trades_ML, start_val= start_value, commission= commission, impact=impact, symbol=sym)
    mktsim.gen_plot(df_delta_trades_ML, portvals, "Strategy Learner: Fund vs Benchmark", symbol= sym)
    #indic.bollingerBand(sym="JPM", sd=dt.date(2008,1,1), ed=dt.date(2009,12,31))
