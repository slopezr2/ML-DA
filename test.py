from datamanager import DataManager
from datamanager import Combiner
from deeplearning import CnnSiata
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#Get Raw Data
#Only load pm5.csv by filter_items
dataM = DataManager(path="data/", filter_items=["pm25"])
station3_pm25 = dataM.get_pm25("3")

#pre-process data
n_input_steps = 24*7*2
n_output_steps = 24*3
pre_processor = Combiner()
datax, datay = pre_processor.combine(n_input_steps, n_output_steps, station3_pm25.CONCENTRATION.values)

#Create Model
n_train = 64*100
n_features = 1
X = datax[0:n_train, :]
Y = datay[0:n_train, :]
X = X.reshape((X.shape[0], X.shape[1],n_features))
cnnSiata = CnnSiata(n_input_steps,n_features, n_output_steps)

#Fit Model
cnnSiata.model.fit(X, Y, epochs=100, verbose=1)

# demonstrate prediction
x_input = datax[n_train+10, :]
x_input = x_input.reshape((1, n_input_steps, n_features))
yhat = cnnSiata.model.predict(x_input, verbose=1)

# plot
plt.plot(np.arange(0, n_input_steps), datax[n_train, :])
plt.plot(np.arange(n_input_steps,n_input_steps+n_output_steps), yhat[0,:],'r')
plt.plot(np.arange(n_input_steps,n_input_steps+n_output_steps), datay[n_train, :], 'g')
plt.show()


