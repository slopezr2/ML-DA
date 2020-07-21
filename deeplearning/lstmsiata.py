from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

class LstSiata:

    def __init__(self, n_input_steps, n_features, n_output_steps):
        self.model = Sequential()
        self.model.add(LSTM(2000, input_shape=(n_input_steps, n_features)))
        self.model.add(RepeatVector(n_output_steps))
        self.model.add(LSTM(2000, return_sequences=True))
        self.model.add(TimeDistributed(Dense(1000, activation='relu')))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])