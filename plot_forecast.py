from datamanager import DataManager
from datamanager import Combiner
from deeplearning import CnnMeteo
from deeplearning import CnvLstmMeteo
import numpy as np



dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])
station82_pm25 = dataM.get_pm25("84")
station82_t = dataM.get_temperature("82")
station82_w = dataM.get_wind("82")

n_input_steps = 24 * 7 * 2
n_output_steps = 24 * 3
pre_processor = Combiner()
X, y = pre_processor.combine(n_input_steps, n_output_steps, station82_t.Value.values, station82_w.Value.values,
                             station82_pm25.CONCENTRATION.values, station82_pm25.CONCENTRATION.values)


dates=(station82_pm25.Date.values)
dates=np.transpose(dates)

datesx,datesy=pre_processor.create_dataset(dates,look_back=n_input_steps,step_forecast=n_output_steps)
n_train = 9500
n_features = X.shape[2]

mls = [ CnnMeteo(n_input_steps, n_features, n_output_steps, drop=True),
       CnvLstmMeteo(n_input_steps, n_features, n_output_steps, reg=False, drop=False),
       CnvLstmMeteo(n_input_steps, n_features, n_output_steps, reg=True, drop=True),
       ]

mls_label = [ 'CnnMeteo_d',
    'CnvLstmMeteo',
    'CnvLstmMeteo_r_d'
]

y_real = y[n_train + 1, :]
y_real = y_real.reshape((1, y_real.shape[0]))
scalerC = pre_processor.scalers[3]
y_real = scalerC.inverse_transform(y_real)
scalerC = pre_processor.scalers[2]
x_prev=X[n_train , :, 2]
x_prev = x_prev.reshape((1, x_prev.shape[0]))
x_prev=scalerC.inverse_transform(x_prev)

for i in range(len(mls_label)):
    mls[i].load("m_" + mls_label[i] + ".h5")

    # demonstrate prediction
    x_input = X[n_train + 1, :, :]
    x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
    yhat = mls[i].predict(x_input, verbose=1)

    print("predicted")
    
    scalerC = pre_processor.scalers[2]
    yhat = scalerC.inverse_transform(yhat)
    testScore = mls[i].compare(y_real, yhat)
    print('Test Score ' + mls_label[i] + ': %.2f RMSE' % (testScore))
    
    mls[i].plot_forecast(datesx[n_train,:],x_prev[0,:],datesy[n_train],yhat[0,:],y_real[0,:],(mls_label[i] + ': %.2f RMSE' % (testScore)),save=True)
