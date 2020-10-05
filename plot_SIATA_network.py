from datamanager import DataManager
from datamanager import Combiner
from deeplearning import CnnMeteo
from deeplearning import CnvLstmMeteo
from deeplearning import LstmBidireccionalMeteo
from deeplearning import LstmVanillaMeteo
from deeplearning import LstmStackedMeteo
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
from numpy import math
import warnings
import json
import sys
import numpy as np
import tensorflow as tf
import sys


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
n_features=5
#Value to graph a moving average of observations. Just for graphical purposes
window_moving_average=5


days_forecast=1
dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])



n_input_steps = 24 * 7 * semana
n_output_steps=24*3

stations=pd.read_csv('./data/Lista_estaciones.csv',delimiter=',')
stations_SIATA=np.array(stations['PM25'].values).astype('str')
stations_Meteo=np.array(stations['Meteo'].values).astype('str')

mls =  CnnMeteo(n_input_steps, n_features, n_output_steps, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells)  
RMSE_LSTM_ML_2={}
for station in range(len(stations_SIATA)):
#for station in range(1):
    print(stations_SIATA[station])
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
    n_train = 9500+527
    n_features = X.shape[2]
            
    dates=(station_pm25.Date.values)
    dates=np.transpose(dates)
    datesx,datesy=pre_processor.create_dataset(dates,look_back=n_input_steps,step_forecast=n_output_steps)
    file_name="./models/Station"+stations_SIATA[station]+".h5"    
    mls.load(file_name)
    for j in range(days_forecast):
        y_real = y[n_train+ 1 + j*24, :]
        y_real = y_real.reshape((1, y_real.shape[0]))
        scalerC = pre_processor.scalers[2]
        y_real = scalerC.inverse_transform(y_real)
        scalerC = pre_processor.scalers[2]
        x_prev=X[n_train+j*24 , :, 2]
        x_prev = x_prev.reshape((1, x_prev.shape[0]))
        x_prev=scalerC.inverse_transform(x_prev)
        # demonstrate prediction
        x_input = X[n_train + 1 + j*24, :, :]
        x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
        yhat = mls.predict(x_input, verbose=1)		
        print("predicted")
        			    
        scalerC = pre_processor.scalers[5]
        yhat = scalerC.inverse_transform(yhat)
        testScore = mls.compare(y_real, yhat)
        print('Test Score ' + "Station"+stations_SIATA[station] + ': %.2f RMSE' % (testScore))
        yreal_moving_average=pd.Series(y_real[0,:]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values				    
        x_prev_moving_average=pd.Series(x_prev[0,:]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values				    
        mls.plot_forecast(datesx[n_train+j*24,:],x_prev_moving_average,datesy[n_train+j*24],yhat[0,:],yreal_moving_average,("Station"+stations_SIATA[station] +'_day_%.1i' % (j+1) ),save=True)
        RMSE_LSTM_ML_2[stations_SIATA[station]]=testScore