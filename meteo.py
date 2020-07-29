from datamanager import DataManager
from datamanager import Combiner
from deeplearning import CnnMeteo
from deeplearning import LstmBidireccionalMeteo
from deeplearning import LstmVanillaMeteo
from deeplearning import LstmStackedMeteo
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
from numpy import math
import warnings
import json

import tensorflow as tf

#Comment to run with GPU o Select CPU

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

# pre-process data
n_input_steps = 24 * 7 * 2
n_output_steps = 24 * 3
pre_processor = Combiner()
X, y = pre_processor.combine(n_input_steps, n_output_steps, station82_t.Value.values, station82_w.Value.values,
                                     station82_pm25.CONCENTRATION.values, station82_pm25.CONCENTRATION.values)


#Create Model
n_train = 9500
n_features = X.shape[2]

mls = [CnnMeteo(n_input_steps,n_features, n_output_steps),LstmVanillaMeteo(n_input_steps,n_features, n_output_steps),
       LstmStackedMeteo(n_input_steps,n_features, n_output_steps),LstmBidireccionalMeteo(n_input_steps,n_features, n_output_steps)  ]
mls_label = ['CnnMeteo', 'LstmVanillaMeteo', 'LstmStackedMeteo', 'LstmBidireccionalMeteo']

for i in range(0,5):
    history = mls[i].model.fit(X, y, epochs=10, verbose=1, batch_size=32)
    hist_df = pd.DataFrame(history.history)
    with open('history/'+mls_label[i], mode='w') as f:
        hist_df.to_json(f)
    mls[i].model.save('models_save/'+mls_label[i])


    # demonstrate prediction
    x_input = X[9500+1,:,:]
    x_input = x_input.reshape((1,x_input.shape[0],x_input.shape[1]))
    yhat = mls[i].model.predict(x_input, verbose=1)

    print("predicted")
    y_real=y[9500+1,:]
    y_real=y_real.reshape((1,y_real.shape[0]))
    scalerC = pre_processor.scalers[2]
    yhat =scalerC.inverse_transform(yhat)
    scalerC = pre_processor.scalers[3]
    yreal = scalerC.inverse_transform(y_real)
    testScore = math.sqrt(mean_squared_error(y_real[0,:], yhat[0,:]))
    print('Test Score '+ mls_label[i] +': %.2f RMSE' % (testScore))
