"Manual_strategy: improved market simulator, accepts trades dataframe instead of csv file"
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt


def compute_portvals(symbols, trades_df, start_val = 100000, commission=9.95, impact=0.005):

    # new market simulator, takes in trade_df (new):
    # a df whose values are net-holdings for ALL days, +1000, -1000, or 0; from start_date to end_date
    dates = trades_df.index

    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices_all.fillna(method = "ffill",inplace=True)
    prices_all.fillna(method = "bfill",inplace=True)
    prices = prices_all[symbols]  # only portfolio symbols
    #print prices

    _cash_price = pd.DataFrame(data= np.ones(len(dates)), index= dates, columns= ['_Cash'])
    prices_df = prices.join(_cash_price)
    #print prices_df

    _cash_amount = pd.DataFrame(data= np.zeros(len(dates)), index= dates, columns= ['_Cash'])
    #now change trades_df to trades_delta_df
    trades_delta_df = trades_df.copy()
    trades_delta_df = trades_delta_df.join(_cash_amount)

    for i in range(0,len(trades_df)):
        if i == 0:
            trades_delta_df.iloc[i,0] = trades_df.iloc[i,0]
        else:
            trades_delta_df.iloc[i,0] = trades_df.iloc[i,0]- trades_df.iloc[i-1,0]
    #print trades_delta_df

    for my_date, row in trades_df.iterrows():
        my_symbol = symbols
        my_share = trades_delta_df[my_symbol[0]][my_date]

        unit_price = prices_df[my_symbol[0]][my_date]
        trades_delta_df.loc[my_date, '_Cash'] += -my_share* unit_price
        #trades_delta_df.loc[my_date,'_Cash'] += -my_share* unit_price
        # commission fees
        if my_share != 0:
            trades_delta_df.loc[my_date,'_Cash'] -= (commission + abs(my_share)* unit_price*impact)

    #initialize
    holdings_df = trades_df.join(_cash_amount)
    holdings_df.set_value(dates[0], '_Cash', start_val)
    # update the 1st row
    holdings_df.iloc[0, 1] += trades_delta_df.iloc[0, 1]
    # update the rest of rows in holdings_df '_Cash' column
    for i in range(1,len(holdings_df.index)):
        holdings_df.iloc[i, 1] += (holdings_df.iloc[i-1, 1] + trades_delta_df.iloc[i, 1])
    #print holdings_df

    values_df = pd.DataFrame(holdings_df.values * prices_df.values, columns=prices_df.columns, index=prices_df.index)
    portvals = values_df.sum(axis = 1)
    #print values_df
    portvals_df = portvals.to_frame()
    return portvals_df

def author():
    return 'yzhu319'

def gen_plot(trades_df, portvals, chart_title):
# take in portvals as dataFrame and compare with SPY in a single chart

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
        #print portvals
        #print portvals.values
    else:
        "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(['JPM'], dates)  # automatically adds SPY
    #prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_JPM = prices_all['JPM']

    BEN_val = 100000 -1000*prices_JPM[0] + 1000*prices_JPM # Bench mark portfolio, cash + 1000 JPM hold
    # Get benchmark values

    BEN_val_ret = BEN_val / BEN_val.shift(1) - 1.0
    BEN_val_ret = BEN_val_ret[1:]

    BEN_val_ev = BEN_val.iloc[-1]
    BEN_val_cr = (BEN_val.iloc[-1] - BEN_val.iloc[0]) /BEN_val.iloc[0]

    BEN_val_adr = (BEN_val_ret - 0).mean()
    BEN_val_sddr = (BEN_val_ret * 1.0).std()
    BEN_val_sr = np.sqrt(252.0) * BEN_val_adr / BEN_val_sddr  # Sharpe Ratio, risk-adjusted return /std of daily return

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # ev, cr, adr, sddr, sr

    portvals_list = portvals.values.tolist()
    portvals_pdSeries = pd.Series(portvals_list)

    port_daily_ret = portvals_pdSeries / portvals_pdSeries.shift(1) - 1.0
    port_daily_ret = port_daily_ret[1:]

    ev = portvals_pdSeries.iloc[-1]
    cr = (portvals_pdSeries.iloc[-1] - portvals_pdSeries.iloc[0]) / portvals_pdSeries.iloc[0]

    adr = (port_daily_ret - 0).mean()
    sddr = (port_daily_ret * 1.0).std()

    sr = np.sqrt(252.0) * adr / sddr  # Sharpe Ratio, risk-adjusted return /std of daily return

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sr)
    print "Sharpe Ratio of Benchmark : {}".format(BEN_val_sr)
    print
    print "Cumulative Return of Fund: {}".format(cr)
    print "Cumulative Return of Benchmark : {}".format(BEN_val_cr)
    print
    print "Standard Deviation of Fund: {}".format(sddr)
    print "Standard Deviation of Benchmark : {}".format(BEN_val_sddr)
    print
    print "Average Daily Return of Fund: {}".format(adr)
    print "Average Daily Return of Benchmark : {}".format(BEN_val_adr)
    print
    print "Final Portfolio Value: {}".format(ev)
    print "Final Portfolio Value of Benchmark : {}".format(BEN_val_ev)
# Compare daily portfolio value with SPY using a normalized plot
    gen_plot = True
    if gen_plot:
        # Normalize
        BEN_val_norm = BEN_val / BEN_val.iloc[0]
        portvals_norm = portvals / portvals.iloc[0]
        fig = plt.figure()
        # retain matplotlib axis object ax
        ax = BEN_val_norm.plot(title= chart_title, label="Benchmark", color = 'b')
        portvals_norm.plot(label="Portf", ax=ax, color ='k')
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(loc="upper left")
        # plot vertical lines for entry points, after a trade; if enter short 1000 pos: RED, if enter long 1000 pos: GREEN
        # filter out those Enter dates

        trades_delta_df = trades_df.copy()
        trades_delta_df[:] = 0
        for i in range(0, len(trades_df)):
            if i == 0:
                trades_delta_df.iloc[i,0] = trades_df.iloc[i,0]
            else:
                trades_delta_df.iloc[i,0] = trades_df.iloc[i,0] - trades_df.iloc[i - 1,0]

        #print trades_df
        #print trades_delta_df

        for i in range(0, len(trades_delta_df)):
            if trades_delta_df.iloc[i,0] != 0:
                if trades_df.iloc[i,0] > 0: # if after the trade, we hold a net long-pos
                    ax.axvline(x=trades_df.index[i], color='g')
                if trades_df.iloc[i,0] < 0:
                    ax.axvline(x=trades_df.index[i], color='r')

        # print trades_delta_df
        #ax.axvline(x = dates[50],color='r')
        fig.savefig('plot_'+chart_title+'.png')
        pass

def test_code():
    trading_data = np.array([0,+1000,-1000,0,+1000])
    dates = np.array(['2009-01-02','2009-01-05','2009-01-06','2009-01-07','2009-01-08'],dtype='datetime64[D]')
    trades_df = pd.DataFrame(data = trading_data, index= dates, columns = ['JPM'])

    portvals = compute_portvals(symbols=["JPM"], trades_df=trades_df, start_val=100000, commission=9.95, impact=0.005)
    gen_plot(trades_df, portvals, "Daily portfolio and SPY")

if __name__ == "__main__":
    test_code()
