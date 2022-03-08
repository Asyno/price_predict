import numpy as np
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from tensorflow.python.keras.models import Sequential


def build_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        hidden_layer: int = 2,
        units: int = 50,
        epochs: int = 100,
        run_number: int = 0
) -> Sequential:
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
