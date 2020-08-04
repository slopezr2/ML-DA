from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import RootMeanSquaredError


class Regularized_ML:
    def __init__(self):
        self.model = Sequential()

    def add_regulization(self, reg, drop):
        if reg:
            self.model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02)))
        else:
            self.model.add(Dense(50, activation='relu'))
        if drop:
            self.model.add(Dropout(0.5))


class CnnMeteo(Regularized_ML):

    def __init__(self, n_input_steps, n_features, n_output_steps, reg=False, drop=False):
        super().__init__()
        self.model.add(LSTM(100, return_sequences=True, input_shape=(n_input_steps, n_features)))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        self.add_regulization(reg, drop)
        self.model.add(Dense(n_output_steps))
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])


class LstmVanillaMeteo(Regularized_ML):

    def __init__(self, n_input_steps, n_features, n_output_steps, reg=False, drop=False):
        super().__init__()
        self.model.add(LSTM(100, input_shape=(n_input_steps, n_features)))
        self.add_regulization(reg, drop)
        self.model.add(Dense(n_output_steps))
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])


class LstmStackedMeteo(Regularized_ML):

    def __init__(self, n_input_steps, n_features, n_output_steps, reg=False, drop=False):
        super().__init__()
        self.model.add(LSTM(100, return_sequences=True, input_shape=(n_input_steps, n_features)))
        self.model.add(LSTM(100))
        self.add_regulization(reg, drop)
        self.model.add(Dense(n_output_steps))
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])


class LstmBidireccionalMeteo(Regularized_ML):

    def __init__(self, n_input_steps, n_features, n_output_steps, reg=False, drop=False):
        super().__init__()
        self.model.add(Bidirectional(LSTM(100, input_shape=(n_input_steps, n_features))))
        self.add_regulization(reg, drop)
        self.model.add(Dense(n_output_steps))
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])


class CnvLstmMeteo:

    def __init__(self, n_input_steps, n_features, n_output_steps):
        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(64, 2, activation='relu'), input_shape=(None, n_input_steps,
                                                                                      n_features)))
        self.model.add(TimeDistributed(MaxPooling1D()))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(n_output_steps))
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])
