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
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error
from numpy import math

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
            
    def fit(self,X, y, epochs=15, batch_size=32, validation_split=0.2, verbose=1):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

    def predict(self,x_input, verbose=1):

        return self.model.predict(x_input, verbose=1)

    def compare(self,y_real,yhat):
        testScore = math.sqrt(mean_squared_error(y_real[0, :], yhat[0,:]))
        return testScore


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

    def __init__(self, n_input_steps, n_features, n_output_steps,reg=False, drop=False):
        self.model = Sequential()
        self.model.add(ConvLSTM2D(64, (1, 3), activation='relu', input_shape= (1,1,n_input_steps, n_features)))
        self.model.add(Flatten())
        self.model.add(RepeatVector(n_output_steps))
        self.model.add(LSTM(200, activation='relu', return_sequences=True))
        self.add_regulization(reg,drop)
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])

    def add_regulization(self, reg, drop):
        if reg:

            self.model.add(TimeDistributed(Dense(100, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02))))
        else:
            self.model.add(TimeDistributed(Dense(100, activation='relu')))
        if drop:
            self.model.add(Dropout(0.5))    
    
    def fit(self,X, y, epochs=15, batch_size=32, validation_split=0.2, verbose=1):
    
        X2=X.reshape((X.shape[0],1 ,1,X.shape[1], X.shape[2])) 
        return self.model.fit(X2, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
    

    
    def predict(self,x_input, verbose=1):
        x_input2=x_input.reshape((1,1 ,1,x_input.shape[1], x_input.shape[2]))
        y=self.model.predict(x_input2, verbose=1)
        return y[0]

    def compare(self,y_real,yhat):
       testScore = math.sqrt(mean_squared_error(y_real[0, :], yhat))
       return testScore
