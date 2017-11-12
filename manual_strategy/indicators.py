
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


def bollingerBand(sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 20
    df = get_data(['JPM'], dates)  # automatically adds SPY
    price_JPM = df['JPM']

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm_JPM = get_rolling_mean(price_JPM, window=window_size)
    # 2. Compute rolling standard deviation
    rstd_JPM = get_rolling_std(price_JPM, window=window_size)
    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_JPM, rstd_JPM)

    # Plot raw JPM values, rolling mean and Bollinger Bands
    ax = price_JPM.plot(title="Bollinger Bands", label='JPM')
    rm_JPM.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left')
    plt.show()

def sma(sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 20
    df = get_data(['JPM'], dates)  # automatically adds SPY
    price_JPM = df['JPM']

    rm_JPM = get_rolling_mean(price_JPM, window=window_size)
    ax = price_JPM.plot(title="Simple Moving Averages", label='JPM')
    rm_JPM.plot(label='SMAs', ax=ax)

    #sma_index = (price_JPM - rm_JPM )/ rm_JPM
    #ax = sma_index.plot(title="SMA indicator", label='price/SMA-1')

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left')
    plt.show()

def ema(sd,ed):
    dates = pd.date_range(sd, ed)
    window_size = 10 #use 10-period window
    df = get_data(['JPM'], dates)  # automatically adds SPY
    price_JPM = df['JPM']

    print price_JPM

    # init ema
    ema_JPM = price_JPM.copy()
    ema_JPM.fillna(0, inplace = True)
    # init the first value
    ema_JPM[window_size] = np.mean(price_JPM[0:window_size])
    # calc the weight factor
    multiplier = 2.0/ (window_size+1)
    # get ema_index[i] from ema_index[i-1]
    for i in range(window_size+1,len(price_JPM)):
        ema_JPM[i] = ema_JPM[i-1] + multiplier *(price_JPM[i] - ema_JPM[i-1])

    ax = price_JPM.plot(title="Exponential Moving Averages", label='JPM')
    ema_JPM.plot(label='EMAs', ax=ax)

    #ema_index = (price_JPM - ema_JPM )/ ema_JPM
    #ax = ema_index.plot(title="EMA indicator", label='price/EMA-1')

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower left')
    plt.show()

    print ema_JPM

if __name__ == '__main__':
    sd = dt.date(2008,1,1)
    ed = dt.date(2009,12,31)
    #bollingerBand(sd,ed)
    #sma(sd,ed)
    ema(sd, ed)