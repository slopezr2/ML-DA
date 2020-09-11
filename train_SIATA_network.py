from datamanager import DataManager
from datamanager import Combiner
from deeplearning import CnnMeteo
from deeplearning import CnvLstmMeteo
from deeplearning import LstmBidireccionalMeteo
from deeplearning import LstmVanillaMeteo
from deeplearning import LstmStackedMeteo
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from numpy import math
import warnings
import json
import sys
import tensorflow as tf

param = int(sys.argv[1])
param=min(param,2)
param=max(param,0)
print(param)
# Comment to run with GPU o Select CPU

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Parameters to select the LSTM architecture

semana=2
n_LSTM_hidden_layers=2
n_cells=200


dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])



n_input_steps = 24 * 7 * semana
n_output_steps=24*3

stations=pd.read_csv('./data/Lista_estaciones.csv',delimiter=',')
stations_SIATA=np.array(stations['PM25'].values).astype('str')
stations_Meteo=np.array(stations['Meteo'].values).astype('str')

for station in range(param*7, (param+1)*7-1):
    if station>18:
        continue
    station_pm25 = dataM.get_pm25(stations_SIATA[station])
    station_t = dataM.get_temperature(stations_Meteo[station])
    station_w = dataM.get_wind(stations_Meteo[station])
    number_samples=min(len(station_t.Value.values),len(station_pm25.CONCENTRATION.values))
    station_pm25 =station_pm25[0:number_samples]
    station_t =station_t[0:number_samples]
    station_w =station_w[0:number_samples]
    pre_processor = Combiner()
    X, y = pre_processor.combine(n_input_steps, n_output_steps, station_t.Value.values, station_w.Value.values,
                                 station_pm25.CONCENTRATION.values,station_t.Date.dt.dayofweek.values,station_t.Date.dt.hour.values , station_pm25.CONCENTRATION.values)
    
    # Create Model
    n_train = 9500
    n_features = X.shape[2]
    
    dates=(station_pm25.Date.values)
    dates=np.transpose(dates)
    datesx,datesy=pre_processor.create_dataset(dates,look_back=n_input_steps,step_forecast=n_output_steps)

    
    mls =  CnnMeteo(n_input_steps, n_features, n_output_steps, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells)

    history = mls.fit(X, y, epochs=300, batch_size=32, validation_split=0.2, verbose=1)
    hist_df = pd.DataFrame(history.history)
    with open("h_Station"+stations_SIATA[station], mode='w') as f:
        hist_df.to_json(f)
    mls.model.save("Station"+stations_SIATA[station]+".h5")
    
    # demonstrate prediction
    x_input = X[n_train + 1, :, :]
    x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
    yhat = mls.predict(x_input, verbose=1)
    
    print("predicted")
    y_real = y[n_train + 1, :]
    y_real = y_real.reshape((1, y_real.shape[0]))
    scalerC = pre_processor.scalers[2]
    yhat = scalerC.inverse_transform(yhat)
    scalerC = pre_processor.scalers[3]
    y_real = scalerC.inverse_transform(y_real)
    testScore = mls.compare(y_real, yhat)
    print("Test Score Station"+stations_SIATA[station]+ " : %.2f RMSE" % (testScore))

    
