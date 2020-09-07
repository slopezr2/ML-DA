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

param = int(sys.argv[1])
print(param)


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])
station82_pm25 = dataM.get_pm25("84")
station82_t = dataM.get_temperature("82")
station82_w = dataM.get_wind("82")





mls_label = [ 'CnnMeteo_d',
    'CnvLstmMeteo',
    'CnvLstmMeteo_r_d'
]


days_forecast=1



#Parameters Tunning experiments 
semanas=[2]
LSTM_hidden_layers=[2]
cells=[200]

for semana in semanas:
	n_input_steps = 24 * 7 * semana
	n_output_steps = 24 * 3
	pre_processor = Combiner()
	X, y = pre_processor.combine(n_input_steps, n_output_steps, station82_t.Value.values, station82_w.Value.values,
                             station82_pm25.CONCENTRATION.values,station82_t.Date.dt.dayofweek.values,station82_t.Date.dt.hour.values , station82_pm25.CONCENTRATION.values)


	dates=(station82_pm25.Date.values)
	dates=np.transpose(dates)

	datesx,datesy=pre_processor.create_dataset(dates,look_back=n_input_steps,step_forecast=n_output_steps)
	n_train = 9500
	n_features = X.shape[2]
	for n_LSTM_hidden_layers in LSTM_hidden_layers:
			for n_cells in cells:
				file_name="m_Weeks_" +str(semana)+"_hidden_"+str(n_LSTM_hidden_layers)+"_cells_"+str(n_cells)+"_"+ mls_label[param] + ".h5"
				titulo=mls_label[param] +str(semana)+"_hidden_"+str(n_LSTM_hidden_layers)+"_cells_"+str(n_cells)
				mls = [ CnnMeteo(n_input_steps, n_features, n_output_steps, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
						 CnvLstmMeteo(n_input_steps, n_features, n_output_steps, reg=False, drop=False,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
						 CnvLstmMeteo(n_input_steps, n_features, n_output_steps, reg=True, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
						 ]
				mls[param].load(file_name)
				for j in range(days_forecast):
				    y_real = y[n_train + j*24, :]
				    y_real = y_real.reshape((1, y_real.shape[0]))
				    scalerC = pre_processor.scalers[2]
				    y_real = scalerC.inverse_transform(y_real)
				    scalerC = pre_processor.scalers[2]
				    x_prev=X[n_train+j*24 , :, 2]
				    x_prev = x_prev.reshape((1, x_prev.shape[0]))
				    x_prev=scalerC.inverse_transform(x_prev)
				    # demonstrate prediction
				    x_input = X[n_train + 1 + j, :, :]
				    x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
				    yhat = mls[param].predict(x_input, verbose=1)		
				    print("predicted")
				    
				    scalerC = pre_processor.scalers[2]
				    yhat = scalerC.inverse_transform(yhat)
				    testScore = mls[param].compare(y_real, yhat)
				    print('Test Score ' + titulo + ': %.2f RMSE' % (testScore))
				    
				    mls[param].plot_forecast(datesx[n_train+j*24,:],x_prev[0,:],datesy[n_train+j*24],yhat[0,:],y_real[0,:],(titulo + ': %.2f RMSE' % (testScore) +' Forecast day %.1i' % (j+1) ),save=True)
