# PYTHONPATH=..:. python grade_strategy_learner2.py

import datetime as dt
import numpy as np
import pandas as pd

import time
import util
import random

import StrategyLearner as sl

in_sample = (dt.datetime(2008,1,1), dt.datetime(2009,12,31))
out_sample = (dt.datetime(2010,1,1), dt.datetime(2011,12,31))
sv = 100000

def additional_test_1(symbol = 'NKE'):
 learner = sl.StrategyLearner(impact=0.0)
 learner.addEvidence(symbol = symbol, sd = in_sample[0], ed = in_sample[1], sv = sv)
 trades = learner.testPolicy(symbol = symbol, sd = in_sample[0], ed = in_sample[1], sv = sv)
 cr = evalPolicy2(symbol, trades, sv, in_sample[0], in_sample[1], 0.005, 0.0)
 benchmark = compute_benchmark(in_sample[0], in_sample[1], sv, symbol, 0.005, 0.0, MAX_HOLDINGS)
 print ''
 print 'Witheld test case 1: In sample test case for an unknown symbol.'
 print 'testPolicy() returns an in-sample result with cumulative return greater than benchmark'
 print 'symbol={} CR={} benchmark={}'.format(symbol, cr, benchmark)
 if cr > benchmark: print 'PASSED'
 else: print 'FAILED'

def get_numbers_of_trades(trades):
 n = trades.shape[0]
 longs = 0
 shorts = 0
 for i in range(n):
  value = trades.ix[i, 0]
  if value > 0.0: longs += 1
  if value < 0.0: shorts += 1
 return (longs, shorts)

def additional_test_2(symbol = 'NKE', impact1 = 0.0, impact2 = 0.10):
 learner1 = sl.StrategyLearner(impact = impact1)
 learner1.addEvidence(symbol = symbol, sd = in_sample[0], ed = in_sample[1], sv = sv)
 trades1 = learner1.testPolicy(symbol = symbol, sd = in_sample[0], ed = in_sample[1], sv = sv)
 learner2 = sl.StrategyLearner(impact = impact2)
 learner2.addEvidence(symbol = symbol, sd = in_sample[0], ed = in_sample[1], sv = sv)
 trades2 = learner2.testPolicy(symbol = symbol, sd = in_sample[0], ed = in_sample[1], sv = sv)
 (l1, s1) = get_numbers_of_trades(trades1)
 (l2, s2) = get_numbers_of_trades(trades2)
 print ''
 print 'Withheld test case 2: In sample test case to verify that strategy accounts for different values of impact'
 print 'Learner returns different trades when impact value is significantly different'
 print 'symbol={} [impact1={}, longs={}, shorts={}] [impact2={}, longs={}, shorts={}]'.format(symbol, impact1, l1, s1, impact2, l2, s2)
 if l1 == l2 and s1 == s2: print 'FAILED'
 else: print 'PASSED'

def main():
 additional_test_1(symbol = 'NKE')
 additional_test_1(symbol = 'AMZN')
 additional_test_2(impact1 = 0.0, impact2 = 0.10)
 additional_test_2(impact1 = -0.5, impact2 = 0.5)
 print ''

# Code copied from the original grade_strategy_learner.py

MAX_HOLDINGS = 1000

def compute_benchmark(sd,ed,sv,symbol,market_impact,commission_cost,max_holdings):
 date_idx = util.get_data([symbol,],pd.date_range(sd,ed)).index
 orders = pd.DataFrame(index=date_idx)
 orders['orders'] = 0; orders['orders'][0] = max_holdings; orders['orders'][-1] = -max_holdings
 return evalPolicy2(symbol,orders,sv,sd,ed,market_impact,commission_cost)

def evalPolicy(student_trades,sym_prices,startval):
 ending_cash = startval - student_trades.mul(sym_prices,axis=0).sum()
 ending_stocks = student_trades.sum()*sym_prices.ix[-1]
 return float((ending_cash+ending_stocks)/startval)-1.0

def evalPolicy2(symbol, student_trades, startval, sd, ed, market_impact,commission_cost):
 orders_df = pd.DataFrame(columns=['Shares','Order','Symbol'])
 for row_idx in student_trades.index:
  nshares = student_trades.loc[row_idx][0]
  if nshares == 0:
   continue
  order = 'sell' if nshares < 0 else 'buy'
  new_row = pd.DataFrame([[abs(nshares),order,symbol],],columns=['Shares','Order','Symbol'],index=[row_idx,])
  orders_df = orders_df.append(new_row)
 portvals = compute_portvals(orders_df, sd, ed, startval,market_impact,commission_cost)
 return float(portvals[-1]/portvals[0])-1

def compute_portvals(orders_df, start_date, end_date, startval, market_impact=0.0, commission_cost=0.0):
 """Simulate the market for the given date range and orders file."""
 symbols = []
 orders = []
 orders_df = orders_df.sort_index()
 for date, order in orders_df.iterrows():
  shares = order['Shares']
  action = order['Order']
  symbol = order['Symbol']
  if action.lower() == 'sell':
   shares *= -1
  order = (date, symbol, shares)
  orders.append(order)
  symbols.append(symbol)
 symbols = list(set(symbols))
 dates = pd.date_range(start_date, end_date)
 prices_all = util.get_data(symbols, dates)
 prices = prices_all[symbols]
 prices = prices.fillna(method='ffill').fillna(method='bfill')
 prices['_CASH'] = 1.0
 trades = pd.DataFrame(index=prices.index, columns=symbols)
 trades = trades.fillna(0)
 cash = pd.Series(index=prices.index)
 cash = cash.fillna(0)
 cash.ix[0] = startval
 for date, symbol, shares in orders:
  price = prices[symbol][date]
  val = shares * price
  # transaction cost model
  val += commission_cost + (pd.np.abs(shares)*price*market_impact)
  positions = prices.ix[date] * trades.sum()
  totalcash = cash.sum()
  if (date < prices.index.min()) or (date > prices.index.max()):
   continue
  trades[symbol][date] += shares
  cash[date] -= val
 trades['_CASH'] = cash
 holdings = trades.cumsum()
 df_portvals = (prices * holdings).sum(axis=1)
 return df_portvals

if __name__ == "__main__":
 main()
