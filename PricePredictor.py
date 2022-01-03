import os
import pickle
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from tensorflow.python.keras.models import Sequential
import PandasUtils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
csv_file_path = "C:/Users/Jan/Zorro/Data/asynML2_trainData_EURUSD_L.csv"
train_data_file_path = "resources/trainData"
model_file_path = "resources/ml/pricePredict"


def get_labeled_data() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if not path.exists(train_data_file_path):
        print("prepare training data")
        data: pd.DataFrame = PandasUtils.get_data_from_csv(csv_file_path, ignore_rows=100)
        print("transform timestamp to sin/cos")
        data = PandasUtils.transform_timestamp(data)
        print("create features and labels")
        pickle.dump(data, open(train_data_file_path, 'wb'))
    else:
        print("load train data")
        data = pickle.load(open(train_data_file_path, 'rb'))

    print("scale transform for the features")
    train_data: pd.DataFrame = data[:int(data.shape[0] * 0.8)]
    test_data: pd.DataFrame = data[int(data.shape[0] * 0.8):]
    sc = MinMaxScaler(feature_range=(0, 1))
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    x_train, y_train = PandasUtils.get_time_range_labels_and_features(train_data, target=10)
    x_test, y_test = PandasUtils.get_time_range_labels_and_features(test_data, target=10)
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
        for i in range(0, hidden_layer):
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
    if not path.exists("resources"):
        os.mkdir("resources")
        os.mkdir("resources/ml")
        os.mkdir("resources/result")
    # units: int = 50
    # hidden_layer: int = 2
    epochs: int = 3
    x_train, y_train, x_test, y_test = get_labeled_data()
    run_number = 3
    for hidden_layer in range(4, 10, 2):
        for units in range(100, 300, 100):
            predictor = build_model(
                x_train,
                y_train,
                units=units,
                hidden_layer=hidden_layer,
                epochs=epochs,
                run_number=run_number
            )
            print("start test prediction")
            test_predict = predictor.predict(x_test)
            # plot result
            plt.plot(y_test[1000:3000], color='red', label='Real Stock Price')
            plt.plot(test_predict[1000:3000], color='blue', label='Predicted Stock Price')
            plt.title('Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.savefig("resources/result/" + str(run_number) + ".png")
            plt.clf()

            result: int = 0
            for row in range(test_predict.shape[0]):
                if test_predict[row] - test_predict[row - 1] > 0 == y_test[row] - y_test[row - 1]:
                    result += 1

            result_file = open("resources/result/result.txt", "a")
            result_file.write(
                str(run_number) +
                " - units: " + str(units) +
                " - hidden layers: " + str(hidden_layer) +
                " - epochs " + str(epochs) +
                " - result " + str(result) +
                "\n"
            )
            result_file.close()
            print("finish module with units: " + str(units) +
                  " - hidden layers: " + str(hidden_layer) +
                  " - epochs " + str(epochs) +
                  " - result " + str(result)
                  )
            run_number += 1
