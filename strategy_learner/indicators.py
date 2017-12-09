"""
@Author: Yuanzheng Zhu (yzhu319)
indicator class to be used in StrategyLearner
Modified based on ManualStrategy project
Input: ticker (sym as a string like "JPM"), start date, end date
Output: indicator value, as data structure as price_JPM
"""


import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt



def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def get_rolling_mean(pd_series, window):
    """Return rolling mean of given values, using specified window size."""
    return pd_series.rolling(window=window, center=False).mean()

def get_rolling_std(pd_series, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd_series.rolling(window=window, center=False).std()

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + 2*rstd
    lower_band = rm - 2*rstd
    return upper_band, lower_band


def bollingerBand(sym, sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 20
    df = get_data([sym], dates)  # automatically adds SPY
    price_JPM = df[sym]

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm = get_rolling_mean(price_JPM, window=window_size)
    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(price_JPM, window=window_size)
    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    # Plot raw JPM values, rolling mean and Bollinger Bands
    ax = price_JPM.plot(title="Bollinger Bands", label=sym)
    rm.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left')
    plt.show()

def sma(sym, sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 20
    df = get_data([sym], dates)  # automatically adds SPY
    price = df[sym]

    sma = get_rolling_mean(price, window=window_size)
    ax = price.plot(title="Simple Moving Averages", label=sym)
    sma.plot(label='SMAs', ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left')
    plt.show()

def ema(sym, sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 10 #use 10-period window
    df = get_data([sym], dates)  # automatically adds SPY
    price = df[sym]

    print price

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

    ax = price.plot(title="Exponential Moving Averages", label=sym)
    ema.plot(label='EMAs', ax=ax)

    #ema_index = (price_JPM - ema_JPM )/ ema_JPM
    #ax = ema_index.plot(title="EMA indicator", label='price/EMA-1')

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left')
    plt.show()

    print ema

if __name__ == '__main__':
    sd = dt.date(2008,1,1)
    ed = dt.date(2009,12,31)
    #bollingerBand(sd,ed)
    #sma(sd,ed)
    ema(sd, ed)
