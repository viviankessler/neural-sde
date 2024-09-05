import pandas as pd
import numpy as np
from datetime import datetime, timedelta

FORMATS = ["%d-%B-%Y", "%Y-%m-%d"]


def load_stock_prices(stock_name="nestle"):
    stock_prices = pd.read_csv("data/" + stock_name + ".csv")[["Date", "Close Price"]]
    if stock_name != "apple":
        stock_prices = stock_prices.iloc[::-1]
    stock_prices = stock_prices.set_index("Date")
    return stock_prices


def get_log_returns(stock_prices, t=5):
    
    date_to_price = {}
    for datestr, price in stock_prices["Close Price"].items():
        for format in FORMATS:
            try:
                date = datetime.strptime(datestr, format)
                if date > datetime(2010, 1, 1):
                    date_to_price[date.date()] = price
                break
            except ValueError:
                continue
    
    log_returns, inital_prices = [], []
    for date in date_to_price:
        if date + timedelta(days=t) in date_to_price:
            log_return = np.log(date_to_price[date + timedelta(days=t)] / date_to_price[date])
            log_returns.append(log_return)
            inital_prices.append(date_to_price[date])
    
    return log_returns, inital_prices
