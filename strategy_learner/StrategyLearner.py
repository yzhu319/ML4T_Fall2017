"""
@Author: Yuanzheng Zhu (yzhu319)
Implementing StrategyLearner
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as util
import math
import BagLearner as bl
import RTLearner as rtl
import ManualStrategy as manu

import random

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = bl.BagLearner(rtl.RTLearner, kwargs={"leaf_size":6}, bags=20, boost=False, verbose=False)

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000):

        # add your code to do learning here

        # example usage of the old backward compatible util function
        # syms=[symbol]
        # dates = pd.date_range(sd, ed)
        # prices_all = util.get_data(syms, dates)  # automatically adds SPY
        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print prices
        #
        # # example use with new colname
        # volume_all = util.get_data(syms, dates, colname ="Volume")  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print volume

        data4train = manu.training_data(symbol, sd, ed, self.impact)

        #convert to np array
        data4train = data4train.as_matrix()
        trainX = data4train[:,0:-1]
        trainY = data4train[:,-1]

        # create a learner and train it; API for BagLearner
        #self.learner = bl.BagLearner(rtl.RTLearner, kwargs={"leaf_size":7}, bags=10, boost=False, verbose=False)  # a Random Forest learner
        self.learner.addEvidence(trainX, trainY)  # train it
        # evaluate in sample
        predY = self.learner.query(trainX)  # get the predictions


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        dates = pd.date_range(sd, ed)
        prices_all = util.get_data([symbol], dates)  # automatically adds SPY
        price = prices_all[symbol]  # only portfolio symbols

        # print price
        data4testX = manu.testing_data(symbol, sd, ed)

        #convert to np array
        data4testX = data4testX.as_matrix()
        testX = data4testX

        # query our learner, what to do in these days
        predPolicy = self.learner.query(testX)  # get the predictions
        predPolicy = pd.Series(data= predPolicy,index= price.index) #convert to pdSeries
        # trades df based on predPolicy
        trades_df = price.copy()
        trades_df[:] = 0

        # # Strategy based on predPolicy, fill in trades_df

        trades_df[predPolicy == 1] = 1000
        trades_df[predPolicy == -1] = -1000
        #trades_df.plot(title = "trades", label="trades" )
        trades_df = trades_df.to_frame()

        #Convert to trades_delta_df
        trades_delta_df = trades_df.copy()
        for i in range(0, len(trades_df)):
            if i == 0:
                trades_delta_df.iloc[i] = trades_df.iloc[i]
            else:
                trades_delta_df.iloc[i] = trades_df.iloc[i] - trades_df.iloc[i - 1]
        #print type(trades_delta_df) # will be passed into marketSim

        #return trades_df # used in my old marketsimcode
        return trades_delta_df # used in auto-grader

if __name__=="__main__":
    print "One does not simply think up a strategy"



