import yfinance as yf
import pandas as pd
from datetime import date
import random

INTERVAL = "1d"
LOCAL_VOL_WINDOW_SIZE = 5


def get_train_test_data(stock_name, train_start, test_start, prediction_period, train_size=1000, test_size=10):
    
    observed_process = get_observed_process(stock_name, train_start, date.today(), interval=INTERVAL, local_vol_window_size=LOCAL_VOL_WINDOW_SIZE)
    
    test_part = observed_process[observed_process.index.to_pydatetime() >= test_start]
    train_part = observed_process[observed_process.index.to_pydatetime() < test_start]

    train_data, test_data = [*get_windows(train_part, prediction_period)], [*get_windows(test_part, prediction_period)]

    return random.sample(train_data, train_size), random.sample(test_data, test_size)


def get_windows(df, prediction_period):
    for i in range(0, df.shape[0] - prediction_period):
        window = df.iloc[i : i + prediction_period]
        if window.shape[0] == prediction_period:
            yield window


def get_observed_process(stock_name, start, end, interval, local_vol_window_size):
    
    observed_prices = get_S(stock_name, start, end, interval)
    observation_times = get_t(observed_prices)
    time_deltas = get_time_deltas(observation_times, observed_prices)
    local_volatilities = get_nu(time_deltas, observed_prices, local_vol_window_size)
    risk_free_rates = get_r(start, end, interval)

    observed_process = pd.DataFrame({
        "S": observed_prices,
        "t": observation_times,
        "time_deltas": time_deltas,
        "ν": local_volatilities,
        "r": risk_free_rates,
        }, index=observed_prices.index
    ).dropna()

    return observed_process


def get_S(stock_name, start, end, interval):
    return yf.download(stock_name, start=start, end=end, interval=interval)["Adj Close"]


def get_t(observed_prices):
    observation_times = observed_prices.index.astype("int64")  # datetime to number of 1e-9 seconds since 1/1/1970
    observation_times = (observation_times - min(observation_times)) / 1e9 / 24 / 60 / 60
    return observation_times


def get_time_deltas(observation_times, observed_prices):
    return pd.Series(observation_times.diff(), index=observed_prices.index)


def get_nu(time_deltas, observed_prices, local_vol_window_size):
    price_quotients = observed_prices.diff() / observed_prices
    local_volatilities = price_quotients.rolling(local_vol_window_size).var().div(time_deltas).shift(periods=local_vol_window_size)
    return local_volatilities


def get_r(start, end, interval):
    treasury_data = yf.download("CSBGC3.SW", start=start, end=end, interval=interval)["Close"]
    observation_times = get_t(treasury_data)
    time_deltas = get_time_deltas(observation_times, treasury_data)
    return (treasury_data.diff() / treasury_data / time_deltas).mean()
