from get_data import get_observed_process
from datetime import date
import pandas as pd

INTERVAL = "1d"
LOCAL_VOL_WINDOW_SIZE = 5


def get_train_test_data(stock_name, train_start, test_start, prediction_period):
    
    observed_process = get_observed_process(stock_name, train_start, date.today(), interval=INTERVAL, local_vol_window_size=LOCAL_VOL_WINDOW_SIZE)
    
    test_part = observed_process[observed_process.index.to_pydatetime() >= test_start]
    train_part = observed_process[observed_process.index.to_pydatetime() < test_start]

    train_data, test_data = [*get_windows(train_part, prediction_period)], [*get_windows(test_part, prediction_period)]

    return train_data, test_data


def get_windows(df, prediction_period):
    for i in range(0, df.shape[0] - prediction_period, prediction_period):
        window = df.iloc[i : i + prediction_period]
        if window.shape[0] == prediction_period:
            yield window
