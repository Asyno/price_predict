import os

import numpy as np
import pandas as pd
from datetime import datetime

CSV_PATH = 'C:/Users/Jan/Zorro/Data/asynEx_EURUSD_L.csv'


def get_data_from_csv(path: str = CSV_PATH, ignore_rows: int = 50) -> pd.DataFrame:
    csv_path = os.path.join(path)
    data = pd.read_csv(csv_path)
    data = data[ignore_rows:]
    data = data.drop_duplicates()
    return data


def get_column(data: pd.DataFrame, column_number: int) -> pd.DataFrame:
    data = data.iloc[:, column_number]
    return data


def remove_column(data: pd.DataFrame, column_number: int) -> pd.DataFrame:
    column_size = data.shape[1]
    column_numbers = [x for x in range(column_size)]  # list of columns' integer indices
    column_numbers.remove(column_number)
    data = data.iloc[:, column_numbers]
    return data


def transform_timestamp(data: pd.DataFrame, index: int = 0) -> pd.DataFrame:
    time_column = get_column(data, index)
    data = remove_column(data, index)
    dates_sin_month = []
    dates_cos_month = []
    dates_sin_year = []
    dates_cos_year = []
    for row in range(0, len(data)):
        dates_sin_month.append(get_time_as_sin(time_column.iloc[row], 60 * 24 * 30))
        dates_cos_month.append(get_time_as_cos(time_column.iloc[row], 60 * 24 * 30))
        dates_sin_year.append(get_time_as_sin(time_column.iloc[row], 60 * 24 * 365))
        dates_cos_year.append(get_time_as_cos(time_column.iloc[row], 60 * 24 * 365))
    data.insert(0, 0, dates_sin_month, True)
    data.insert(0, 1, dates_cos_month, True)
    data.insert(0, 2, dates_sin_year, True)
    data.insert(0, 3, dates_cos_year, True)
    return data


def get_time_as_sin(date_string: str, time_frame: int) -> int:
    timestamp = datetime.strptime(date_string, '%y%m%d %H:%M:%S').timestamp()
    return np.sin(timestamp * (2 * np.pi / time_frame))


def get_time_as_cos(date_string: str, time_frame: int) -> int:
    timestamp = datetime.strptime(date_string, '%y%m%d %H:%M:%S').timestamp()
    return np.cos(timestamp * (2 * np.pi / time_frame))


def get_time_range_labels_and_features(
        data: pd.DataFrame,
        time_range: int = 60,
        target: int = 0
) -> (np.ndarray, np.ndarray):
    x_train = []
    y_train = []
    for i in range(time_range, len(data)):
        if len(data) > i + target:
            # add all fields from the row as feature
            features = []
            for j in range(0, time_range):
                features.append(data[j, :data.shape[1] - 1])
            x_train.append(features)
            # the last field should be the label
            y_train.append(data[i + target, data.shape[1] - 1])
    return np.array(x_train), np.array(y_train)
