import yfinance as yf
import pandas as pd
import numpy as np
import random



def get_samples(stock_name, start, end, prediction_period, interval, local_window_size, sample_size):
    observed_process = get_observed_process(stock_name, start, end, interval=interval, local_window_size=local_window_size)
    windows = [*get_windows(observed_process, prediction_period)]
    return random.sample(windows, sample_size)


def get_windows(df, prediction_period):
    for i in range(0, df.shape[0] - prediction_period):
        window = df.iloc[i : i + prediction_period]
        if window.shape[0] == prediction_period:
            yield window


def get_observed_process(stock_name, start, end, interval, local_window_size):
    
    observed_prices = get_S(stock_name, start, end, interval)
    observation_times = get_t(observed_prices)
    time_deltas = get_deltas(observation_times)
    price_deltas = get_deltas(observed_prices)
    risk_free_rates = get_rates(price_deltas, observed_prices, time_deltas, local_window_size)
    local_volatilities = get_nu(risk_free_rates, price_deltas, observed_prices, time_deltas, local_window_size)
    local_volatility_deltas = get_deltas(local_volatilities)

    observed_process = pd.DataFrame({
        "t": observation_times,
        "r": risk_free_rates,
        "S": observed_prices,
        "ν": local_volatilities,
        "theta": local_volatilities.rolling(10 * local_window_size, min_periods=local_window_size).mean(),
        "_time_deltas": time_deltas,
        "_price_deltas": price_deltas,
        "_loc_vol_deltas": local_volatility_deltas,
        }, index=observed_prices.index
    ).dropna()

    return observed_process


def get_S(stock_name, start, end, interval):
    return yf.download(stock_name, start=start, end=end, interval=interval)["Adj Close"]


def get_t(observed_prices):
    observation_times = observed_prices.index.astype("int64")  # datetime to number of 1e-9 seconds since 1/1/1970
    observation_times = (observation_times - min(observation_times)) / 1e9 / 24 / 60 / 60
    return pd.Series(observation_times, index=observed_prices.index)


def get_deltas(array):
    return array.diff().shift(periods=-1)


def get_rates(price_deltas, observed_prices, time_deltas, local_window_size):
    return (price_deltas / observed_prices / time_deltas).rolling(local_window_size).mean()


def get_nu(risk_free_rates, price_deltas, observed_prices, time_deltas, local_window_size):
    numerator = price_deltas - risk_free_rates * observed_prices * time_deltas
    denominator = observed_prices * np.sqrt(time_deltas)
    return (numerator / denominator).rolling(local_window_size).var() / local_window_size
