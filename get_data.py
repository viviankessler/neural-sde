import yfinance as yf
import pandas as pd


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
