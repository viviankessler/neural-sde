import yfinance as yf
import pandas as pd
import numpy as np
import random


def get_samples(stock_names, num_samples, start, end, interval, local_window_size, prediction_period):
    observed_processes = [get_observed_process(stock_name, start, end, interval, local_window_size) for stock_name in stock_names]
    windows = [*get_windows(observed_processes, prediction_period)]
    return random.sample(windows, num_samples)


def get_windows(observed_processes, prediction_period):
    for i in range(0, observed_processes[0].shape[0] - prediction_period):
        windows = []
        for observed_process in observed_processes:
            if len(windows) == 0:
                window = observed_process.iloc[i : i + prediction_period]
            else:
                index = windows[0].index
                window = observed_process.loc[index].dropna()
            if window.shape[0] == prediction_period:
                windows.append(window)
        shapes = set(w.shape for w in windows)
        assert len(shapes) == 1, shapes
        yield windows


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
    return array.diff()


def get_rates(price_deltas, observed_prices, time_deltas, local_window_size):
    return (price_deltas / observed_prices.shift(periods=1) / time_deltas).rolling(local_window_size).mean()


def get_nu(risk_free_rates, price_deltas, observed_prices, time_deltas, local_window_size):
    numerator = price_deltas - risk_free_rates * observed_prices.shift(periods=1) * time_deltas
    denominator = observed_prices * np.sqrt(time_deltas)
    return (numerator / denominator).rolling(local_window_size).var() / local_window_size
