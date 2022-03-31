import os
from datetime import datetime
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CSV_PATH = 'C:/Users/Jan/Zorro/Data/asynEx_EURUSD_L.csv'


def get_data_from_csv(path: str = CSV_PATH, ignore_rows: int = 50) -> pd.DataFrame:
    csv_path: str = os.path.join(path)
    data = pd.read_csv(csv_path)
    data = data[ignore_rows:]
    data = data.drop_duplicates()
    return data


def get_column(data: pd.DataFrame, column_number: int) -> pd.DataFrame:
    data = data.iloc[:, column_number]
    return data


def remove_column(data: pd.DataFrame, column_number: int) -> pd.DataFrame:
    column_size: int = data.shape[1]
    column_numbers: List[int] = [x for x in range(column_size)]  # list of columns' integer indices
    column_numbers.remove(column_number)
    data: pd.DataFrame = data.iloc[:, column_numbers]
    return data


def transform_timestamp(data: pd.DataFrame, index: int = 0) -> pd.DataFrame:
    time_column = get_column(data, index)
    data = remove_column(data, index)
    data.insert(0, 0, time_column, True)
    data.insert(0, 1, time_column, True)
    data.insert(0, 2, time_column, True)
    data.insert(0, 3, time_column, True)
    data.apply(lambda x: transform_timestamp_per_row(x), axis=1)
    return data


def transform_timestamp_per_row(row: pd.Series) -> pd.Series:
    row[0] = get_time_as_sin(row[0], 60 * 24 * 30)
    row[1] = get_time_as_cos(row[1], 60 * 24 * 30)
    row[2] = get_time_as_sin(row[2], 60 * 24 * 365)
    row[3] = get_time_as_cos(row[3], 60 * 24 * 365)
    return row


def get_time_as_sin(date_string: str, time_frame: int) -> int:
    timestamp = datetime.strptime(date_string, '%y%m%d %H:%M:%S').timestamp()
    return np.sin(timestamp * (2 * np.pi / time_frame))


def get_time_as_cos(date_string: str, time_frame: int) -> int:
    timestamp = datetime.strptime(date_string, '%y%m%d %H:%M:%S').timestamp()
    return np.cos(timestamp * (2 * np.pi / time_frame))


def get_fibonacci(target: int) -> list:
    fibonacci: list = [1]
    value: int = 2
    while value < target:
        fibonacci.append(value)
        value += fibonacci[len(fibonacci) - 2]
    return fibonacci


class LabelType(Enum):
    FIBONACCI = 1
    FULL = 2
    AUTO = 3


def get_labels_fibonacci_type(data: pd.DataFrame, row_number: int, target: int, time_range: int):
    features = []
    for j in range(row_number - time_range, row_number, target):
        data_range: pd.DataFrame = data[j: j + target]
        features.append(data_range.mean(axis=0))
    if target >= 10:
        fibonacci: list = get_fibonacci(target)
        fibonacci.reverse()
        for f in fibonacci:
            features.append(data[row_number - target + f])
    return features


def get_labels_full_type(data: pd.DataFrame, row_number: int, time_range: int):
    features = []
    for row in range(row_number - time_range, row_number):
        features.append(data[row])
    return features


def get_labels_auto_type(data: pd.DataFrame, row_number: int, target: int, time_range: int):
    if target >= 30:
        return get_labels_fibonacci_type(data, row_number, target, time_range)
    else:
        return get_labels_full_type(data, row_number, time_range)


def get_time_range_labels_and_features(
        data: pd.DataFrame,
        time_range: int = 60,
        target: int = 10,
        label_type: LabelType = LabelType.AUTO
) -> (np.ndarray, np.ndarray):
    x_train = []
    y_train = []
    for i in range(time_range, len(data)):
        if len(data) > i + target:
            # add all fields from the row as feature
            if label_type == LabelType.FIBONACCI:
                x_train.append(get_labels_fibonacci_type(data, i, target, time_range))
            elif label_type == LabelType.FULL:
                x_train.append(get_labels_full_type(data, i, time_range))
            elif label_type == LabelType.AUTO:
                x_train.append(get_labels_auto_type(data, i, target, time_range))
            else:
                raise NotImplementedError
            # the last field should be the label
            # get the mean from the range 'i' to i + target (the target value) + target
            data_range: pd.DataFrame = data[i: i + target + target, data.shape[1] - 1]
            y_train.append(data_range.mean(axis=0))
    return np.array(x_train), np.array(y_train)


def get_labeled_data(
        csv_file_path: str,
        use_small_dataset: bool = False,
        time_range: int = 60,
        target: int = 10,
        label_type: LabelType = LabelType.AUTO
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    print("prepare training data")
    data: pd.DataFrame = get_data_from_csv(csv_file_path, ignore_rows=100)
    if use_small_dataset:
        data = data[int(data.shape[0] * 0.5):]
    print("transform timestamp to sin/cos")
    data = transform_timestamp(data)

    print("split train and test data")
    train_data: pd.DataFrame = data[:int(data.shape[0] * 0.8)]
    test_data: pd.DataFrame = data[int(data.shape[0] * 0.8):]

    print("scale transform for the features")
    sc = MinMaxScaler(feature_range=(0, 1))
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    print("create features and labels for train and test set")
    x_train, y_train = get_time_range_labels_and_features(
        train_data,
        target=target,
        time_range=time_range,
        label_type=label_type
    )
    x_test, y_test = get_time_range_labels_and_features(
        test_data,
        target=target,
        time_range=time_range,
        label_type=label_type
    )
    return x_train, y_train, x_test, y_test
