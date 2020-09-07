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
print(param)
# Comment to run with GPU o Select CPU

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Parameters to select the LSTM architecture
days_forecast=1
semana=2
n_LSTM_hidden_layers=2
n_cells=200
n_features=5

dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])
station82_pm25 = dataM.get_pm25("84")
station82_t = dataM.get_temperature("82")
station82_w = dataM.get_wind("82")

n_size=100
n_input_steps = 24 * 7 * semana
n_output_steps=24*3
data_pm25 = np.copy(station82_pm25.CONCENTRATION.values)
data_t = np.copy(station82_t.Value.values)
data_w = np.copy(station82_w.Value.values)

ensembles_pm25 = dataM.generate_ensambles(data_pm25,2.4,n_size)
ensembles_t = dataM.generate_ensambles(data_t,0.5,n_size)
ensembles_w = dataM.generate_ensambles(data_w,0.05,n_size)

mls_label = [ 'CnnMeteo_d',
    'CnvLstmMeteo',
    'CnvLstmMeteo_r_d'
]

file_name="m_Weeks_" +str(semana)+"_hidden_"+str(n_LSTM_hidden_layers)+"_cells_"+str(n_cells)+"_"+ mls_label[param] + ".h5"
titulo=mls_label[param] +str(semana)+"_hidden_"+str(n_LSTM_hidden_layers)+"_cells_"+str(n_cells)
mls = [ CnnMeteo(n_input_steps, n_features, n_output_steps, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
		 CnvLstmMeteo(n_input_steps, n_features, n_output_steps, reg=False, drop=False,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
    	 CnvLstmMeteo(n_input_steps, n_features, n_output_steps, reg=True, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
						 ]
mls[param].load(file_name)

n_output_steps = 24 * 3
pre_processor = Combiner()
y_real_ensemble=[]
yhat_ensemble=[]
x_prev_ensemble=[]
for i in range(n_size):
    X, y = pre_processor.combine(n_input_steps, n_output_steps, ensembles_t[:,i], ensembles_w[:,i],
                             ensembles_pm25[:,i],station82_t.Date.dt.dayofweek.values,station82_t.Date.dt.hour.values , station82_pm25.CONCENTRATION.values)
    dates=(station82_pm25.Date.values)
    dates=np.transpose(dates)
    datesx,datesy=pre_processor.create_dataset(dates,look_back=n_input_steps,step_forecast=n_output_steps)
    n_train = 9500
    n_features = X.shape[2]
    for j in range(days_forecast):
        y_real = y[n_train + j*24, :]
        y_real = y_real.reshape((1, y_real.shape[0]))
        scalerC = pre_processor.scalers[2]
        y_real = scalerC.inverse_transform(y_real)
        y_real_ensemble.append(y_real)
        scalerC = pre_processor.scalers[2]
        x_prev=X[n_train+j*24 , :, 2]
        x_prev = x_prev.reshape((1, x_prev.shape[0]))
        x_prev=scalerC.inverse_transform(x_prev)
        x_prev_ensemble.append(x_prev)
		# demonstrate prediction
        x_input = X[n_train + 1 + j, :, :]
        x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
        yhat = mls[param].predict(x_input, verbose=1)
        scalerC = pre_processor.scalers[2]
        yhat = scalerC.inverse_transform(yhat)
        yhat_ensemble.append(yhat)


yhat_ensemble=np.array(yhat_ensemble)
yhat_ensemble=yhat_ensemble.reshape((yhat_ensemble.shape[0],yhat_ensemble.shape[2]))

y_real_ensemble=np.array(y_real_ensemble)
y_real_ensemble=y_real_ensemble.reshape((y_real_ensemble.shape[0],y_real_ensemble.shape[2]))

x_prev_ensemble=np.array(x_prev_ensemble)
x_prev_ensemble=x_prev_ensemble.reshape((x_prev_ensemble.shape[0],x_prev_ensemble.shape[2]))

fig, ax = plt.subplots()
ax.plot(datesx[n_train+j*24,:],np.mean(x_prev_ensemble,0),'b',linewidth=2,label='Input Data')
ax.plot(datesy[n_train+j*24,:],np.mean(yhat_ensemble,0),'g',linewidth=2,label='Prediction')
ax.plot(datesy[n_train+j*24,:],np.mean(y_real_ensemble,0),'r*-',linewidth=1,markersize=5,label='Real Data')
ax.fill_between(datesx[n_train+j*24,:],np.mean(x_prev_ensemble,0)- np.std(x_prev_ensemble,0),np.mean(x_prev_ensemble,0)+ np.std(x_prev_ensemble,0),alpha=0.3,color='blue')
ax.fill_between(datesy[n_train+j*24,:],np.mean(yhat_ensemble,0)- np.std(yhat_ensemble,0),np.mean(yhat_ensemble,0)+ np.std(yhat_ensemble,0),alpha=0.3,color='green')
plt.legend(fontsize=20)


xmin, xmax = ax.get_xlim()
custom_ticks = np.linspace(xmin, xmax, 10, dtype=int)
ax.set_xticks(custom_ticks)
plt.tight_layout()

fig, ax = plt.subplots()
sns.distplot(yhat_ensemble[:,-1]-np.mean(yhat_ensemble[:,-1],0), hist=False, kde=True, 
              color = 'green',label='PDF ML observations' ,
             kde_kws={'linewidth': 4})
sns.distplot(x_prev_ensemble[:,-1]-np.mean(x_prev_ensemble[:,-1],0), hist=False, kde=True, 
              color = 'blue',label='PDF SIATA observations' ,
             kde_kws={'linewidth': 4})
plt.legend(fontsize=20)