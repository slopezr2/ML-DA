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

dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])
station82_pm25 = dataM.get_pm25("84")
station82_t = dataM.get_temperature("82")
station82_w = dataM.get_wind("82")

#Parameters Tunning experiments 
semanas=[1,2,3]
LSTM_hidden_layers=[1,2,3]
cells=[100,200,500,1000]

mls_label = [ 'CnnMeteo_d',
	  'CnvLstmMeteo',
	  'CnvLstmMeteo_r_d'
]
for semana in semanas:
	n_input_steps = 24 * 7 * semana
	n_output_steps = 24 * 3
	# pre-process data
	pre_processor = Combiner()
	X, y = pre_processor.combine(n_input_steps, n_output_steps, station82_t.Value.values, station82_w.Value.values,
                             station82_pm25.CONCENTRATION.values,station82_t.Date.dt.dayofweek.values,station82_t.Date.dt.hour.values , station82_pm25.CONCENTRATION.values)

	# Create Model
	n_train = 9500
	n_features = X.shape[2]

	for n_LSTM_hidden_layers in LSTM_hidden_layers:
	
		for n_cells in cells:
		
			mls = [ CnnMeteo(n_input_steps, n_features, n_output_steps, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
						 CnvLstmMeteo(n_input_steps, n_features, n_output_steps, reg=False, drop=False,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
						 CnvLstmMeteo(n_input_steps, n_features, n_output_steps, reg=True, drop=True,n_LSTM_hidden_layers=n_LSTM_hidden_layers,n_cells=n_cells),
						 ]

			history = mls[param].fit(X, y, epochs=300, batch_size=32, validation_split=0.2, verbose=1)
			hist_df = pd.DataFrame(history.history)
			with open("h_input_"+str(semana)+"_hidden_"+str(n_LSTM_hidden_layers)+"_cells_"+str(n_cells)+"_"+ mls_label[param], mode='w') as f:
					hist_df.to_json(f)
			mls[param].model.save("m_" +str(semana)+"_hidden_"+str(n_LSTM_hidden_layers)+"_cells_"+str(n_cells)+"_"+ mls_label[param] + ".h5")

			# demonstrate prediction
			x_input = X[9500 + 1, :, :]
			x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
			yhat = mls[param].predict(x_input, verbose=1)

			print("predicted")
			y_real = y[9500 + 1, :]
			y_real = y_real.reshape((1, y_real.shape[0]))
			scalerC = pre_processor.scalers[2]
			yhat = scalerC.inverse_transform(yhat)
			scalerC = pre_processor.scalers[3]
			y_real = scalerC.inverse_transform(y_real)
		testScore = mls[param].compare(y_real, yhat)
		print('Test Score ' + mls_label[param] +str(semana)+"_hidden_"+str(n_LSTM_hidden_layers)+"_cells_"+str(n_cells)+"_"+ ': %.2f RMSE' % (testScore))
