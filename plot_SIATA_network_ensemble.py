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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import sys
import numpy as np
import tensorflow as tf
import sys
plt.rcParams['text.usetex'] = True

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

n_size=100

days_forecast=3
dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])



n_input_steps = 24 * 7 * semana
n_output_steps=24*3

stations=pd.read_csv('./data/Lista_estaciones.csv',delimiter=',')
stations_SIATA=np.array(stations['PM25'].values).astype('str')
stations_Meteo=np.array(stations['Meteo'].values).astype('str')

mls =  CnnMeteo(n_input_steps, n_features, n_output_steps, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells)  
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
    data_pm25 = np.copy(station_pm25.CONCENTRATION.values)
    data_t = np.copy(station_t.Value.values)
    data_w = np.copy(station_w.Value.values)
    ensembles_pm25 = dataM.generate_ensambles(data_pm25,2.4,n_size)
    ensembles_t = dataM.generate_ensambles(data_t,0.5,n_size)
    ensembles_w = dataM.generate_ensambles(data_w,0.05,n_size)
    y_real_ensemble=[]
    yhat_ensemble=[]
    x_prev_ensemble=[]
    for i in range(n_size):
        X, y = pre_processor.combine(n_input_steps, n_output_steps, ensembles_t[:,i], ensembles_w[:,i],
                                 ensembles_pm25[:,i],station_t.Date.dt.dayofweek.values,station_t.Date.dt.hour.values , station_pm25.CONCENTRATION.values)
              
        # Create Model
        n_train = 9500+527+5
        n_features = X.shape[2]
                
        dates=(station_pm25.Date.values)
        dates=np.transpose(dates)
        datesx,datesy=pre_processor.create_dataset(dates,look_back=n_input_steps,step_forecast=n_output_steps)
        file_name="./models/Station"+stations_SIATA[station]+".h5"    
        mls.load(file_name)
        y_real = y[n_train+1, :]
        y_real = y_real.reshape((1, y_real.shape[0]))
        scalerC = pre_processor.scalers[2]
        y_real = scalerC.inverse_transform(y_real)
        y_real_ensemble.append(y_real)
        scalerC = pre_processor.scalers[2]
        x_prev=X[n_train , :, 2]
        x_prev = x_prev.reshape((1, x_prev.shape[0]))
        x_prev=scalerC.inverse_transform(x_prev)
        x_prev_ensemble.append(x_prev)
        # demonstrate prediction
        x_input = X[n_train + 1, :, :]
        x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
        yhat = mls.predict(x_input, verbose=1)		
        print("predicted")
        scalerC = pre_processor.scalers[5]
        yhat = scalerC.inverse_transform(yhat)
        yhat_ensemble.append(yhat)
        testScore = mls.compare(y_real, yhat)
        print('Test Score ' + "Station"+stations_SIATA[station] + ': %.2f RMSE' % (testScore))
        yreal_moving_average=pd.Series(y_real[0,:]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values				    
        x_prev_moving_average=pd.Series(x_prev[0,:]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values				    
        
    yhat_ensemble=np.array(yhat_ensemble)
    yhat_ensemble=yhat_ensemble.reshape((yhat_ensemble.shape[0],yhat_ensemble.shape[2]))
    
    y_real_ensemble=np.array(y_real_ensemble)
    y_real_ensemble=y_real_ensemble.reshape((y_real_ensemble.shape[0],y_real_ensemble.shape[2]))
    
    x_prev_ensemble=np.array(x_prev_ensemble)
    x_prev_ensemble=x_prev_ensemble.reshape((x_prev_ensemble.shape[0],x_prev_ensemble.shape[2]))

    fig, ax = plt.subplots(figsize=(30,15),dpi=200)
    ax.plot(datesx[n_train+1,:],np.mean(x_prev_ensemble,0),'b',linewidth=2,label='Input Data')
    ax.plot(datesy[n_train+1,:],np.mean(yhat_ensemble,0),'g',linewidth=2,label='Prediction')
    ax.plot(datesy[n_train+1,:],np.mean(y_real_ensemble,0),'r*-',linewidth=2,markersize=10,label='Real Data')
    ax.fill_between(datesx[n_train+1,:],np.mean(x_prev_ensemble,0)- np.std(x_prev_ensemble,0),np.mean(x_prev_ensemble,0)+ np.std(x_prev_ensemble,0),alpha=0.3,color='blue')
    ax.fill_between(datesy[n_train+1,:],np.mean(yhat_ensemble,0)- np.std(yhat_ensemble,0),np.mean(yhat_ensemble,0)+ np.std(yhat_ensemble,0),alpha=0.3,color='green')
    dates=np.concatenate((datesx[n_train+1,:],datesy[n_train+1,:]))
    plt.ylabel('PM$_{2.5}$ Concentration [$\mu$g/m$^3$]',fontsize=45)
    plt.grid(axis='x')
    plt.yticks(fontsize=30)
    plt.legend(fontsize=35)
    ax = plt.gca()
    just_day=lambda t: t[6:10]
    just_days=np.array([just_day(t) for t in dates])
    custom_ticks = np.arange(0, 408, 24)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(just_days[custom_ticks],fontsize=35)
    plt.savefig("./figures/Ensemble_Station"+stations_SIATA[station]+".png")
    
    
    fig, ax = plt.subplots()
    sns.distplot(yhat_ensemble[:,-1]-np.mean(yhat_ensemble[:,-1],0), hist=False, kde=True, 
                  color = 'green',label='PDF ML observations' ,
                 kde_kws={'linewidth': 4})
    sns.distplot(x_prev_ensemble[:,-1]-np.mean(x_prev_ensemble[:,-1],0), hist=False, kde=True, 
                  color = 'blue',label='PDF SIATA observations' ,
                 kde_kws={'linewidth': 4})
    plt.legend(fontsize=20)
    plt.savefig("./figures/Ensemble_Station"+stations_SIATA[station]+".png")