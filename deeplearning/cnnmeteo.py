from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from keras.metrics import RootMeanSquaredError

class CnnMeteo:

    def __init__(self, n_input_steps, n_features, n_output_steps):
        self.model = Sequential()
        self.model.add(LSTM(100, return_sequences=True, input_shape=(n_input_steps, n_features)))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(n_output_steps))
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])