from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

class CnnSiata:

    def __init__(self, n_input_steps, n_features, n_output_steps):
        self.model = Sequential()
        self.model.add(Conv1D(64, 2, activation='relu', input_shape=(n_input_steps,n_features)))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(n_output_steps))
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])