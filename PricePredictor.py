import os
import pickle
import time
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from tensorflow.python.keras.models import Sequential

from price_predict.utils import PandasUtils
from price_predict.utils.PandasUtils import LabelType

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
csv_file_path = "resources/train_data/asynML2_trainData_EURUSD_L.csv"
train_data_file_path = "resources/train_data/trainData"
model_file_path = "resources/ml/pricePredict"
label_version: str = 'V5'


def create_resources_folder_if_not_exists():
    if not path.exists("resources"):
        os.mkdir("resources")
        os.mkdir("resources/ml")
        os.mkdir("resources/result")
        os.mkdir("resources/train_data")


def get_labeled_data(
        use_existing: bool = True,
        use_small_dataset: bool = False,
        time_range: int = 60,
        target: int = 10,
        label_type: LabelType = LabelType.AUTO
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if not use_existing or not path.exists(train_data_file_path):
        print("prepare training data")
        data: pd.DataFrame = PandasUtils.get_data_from_csv(csv_file_path, ignore_rows=100)
        if use_small_dataset:
            data = data[int(data.shape[0] * 0.5):]
        print("transform timestamp to sin/cos")
        data = PandasUtils.transform_timestamp(data)
        # TODO: should add information about the settings to the file name
        pickle.dump(data, open(train_data_file_path, 'wb'))
    else:
        print("load train data")
        data = pickle.load(open(train_data_file_path, 'rb'))

    print("split train and test data")
    train_data: pd.DataFrame = data[:int(data.shape[0] * 0.8)]
    test_data: pd.DataFrame = data[int(data.shape[0] * 0.8):]

    print("scale transform for the features")
    sc = MinMaxScaler(feature_range=(0, 1))
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    print("create features and labels for train and test set")
    x_train, y_train = PandasUtils.get_time_range_labels_and_features(
        train_data,
        target=target,
        time_range=time_range,
        label_type=label_type
    )
    x_test, y_test = PandasUtils.get_time_range_labels_and_features(
        test_data,
        target=target,
        time_range=time_range,
        label_type=label_type
    )
    return x_train, y_train, x_test, y_test


def build_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        hidden_layer: int = 2,
        units: int = 50,
        epochs: int = 100,
        run_number: int = 0
) -> Sequential:
    if not path.exists(model_file_path + str(run_number)):
        print("build model")
        regressor = Sequential()
        regressor.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        regressor.add(Dropout(0.2))
        for _ in range(0, hidden_layer):
            regressor.add(LSTM(units=units, return_sequences=True))
            regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=units))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        print("train model")
        regressor.fit(x_train, y_train, epochs=epochs, batch_size=32)
        print("save model")
        regressor.save(model_file_path + str(run_number))
    else:
        print("found and load model")
        regressor = tf.keras.models.load_model(model_file_path + str(run_number))
    return regressor


class Test:
    create_resources_folder_if_not_exists()

    # TODO: create a config file for this parameter
    units_params: list = [50]
    hidden_layer_params: list = [5]
    epochs: int = 10
    target_params: list = [10, 30, 60]
    use_small_dataset: bool = True
    time_range_multiplier: int = 10
    label_type: LabelType = LabelType.FULL

    for hidden_layer in hidden_layer_params:
        for units in units_params:
            for target in target_params:
                run_number: int = len(os.listdir("resources/ml"))
                run_message: str = (
                        str(run_number) +
                        " - units: " + str(units) +
                        " - hidden layers: " + str(hidden_layer) +
                        " - epochs " + str(epochs) +
                        " - use small data: " + str(use_small_dataset) +
                        " - target: " + str(target) +
                        " - label type: " + str(label_type.name) +
                        " - label version: " + label_version
                )

                print("build module with run number: " + run_message)

                x_train, y_train, x_test, y_test = get_labeled_data(
                    use_existing=False,
                    use_small_dataset=use_small_dataset,
                    time_range=target * time_range_multiplier,
                    target=target,
                    label_type=label_type
                )

                module_build_start_time: float = time.time()
                predictor = build_model(
                    x_train,
                    y_train,
                    units=units,
                    hidden_layer=hidden_layer,
                    epochs=epochs,
                    run_number=run_number
                )
                module_build_end_time: float = time.time()

                print("start test prediction")
                test_predict = predictor.predict(x_test)
                # plot result
                plt.plot(y_test[1000 + target:3000 + target], color='red', label='Real Stock Price')
                plt.plot(test_predict[1000:3000], color='blue', label='Predicted Stock Price')
                plt.title('Stock Price Prediction')
                plt.xlabel('Time')
                plt.ylabel('Stock Price')
                plt.legend()
                plt.savefig("resources/result/" + str(run_number) + ".png")
                plt.clf()

                result: int = 0
                for row in range(test_predict.shape[0]):
                    if (test_predict[row] - test_predict[row - 1] > 0) == (y_test[row] - y_test[row - 1] > 0):
                        result += 1

                result_message: str = run_message + " - result " + str(result)
                build_time: float = module_build_end_time - module_build_start_time
                result_message += " - module train time in min: " + str(build_time / 60)
                result_file = open("resources/result/result.txt", "a")
                result_file.write(result_message + "\n")
                result_file.close()
                print("finish module with run number: " + result_message)
