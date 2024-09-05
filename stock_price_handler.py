import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_stock_prices(path="nestle.csv"):
    stock_prices = pd.read_csv(path)[["Date", "Close Price"]]
    stock_prices = stock_prices.iloc[::-1]
    stock_prices = stock_prices.set_index("Date")
    return stock_prices


def get_log_returns(stock_prices, t=5):
    
    date_to_price = {}
    for datestr, price in stock_prices["Close Price"].items():
        date_to_price[datetime.strptime(datestr, "%d-%B-%Y").date()] = price
    
    log_returns = []
    for date in date_to_price:
        if date + timedelta(days=t) in date_to_price:
            log_return = np.log(date_to_price[date + timedelta(days=t)] / date_to_price[date])
            log_returns.append(log_return)
    
    return log_returns
