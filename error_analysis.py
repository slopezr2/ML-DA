from datamanager import DataManager
from datamanager import Combiner
from deeplearning import CnnMeteo
from deeplearning import CnvLstmMeteo
from deeplearning import LstmBidireccionalMeteo
from deeplearning import LstmVanillaMeteo
from deeplearning import LstmStackedMeteo
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
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

dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])
station82_pm25 = dataM.get_pm25("84")
station82_t = dataM.get_temperature("82")
station82_w = dataM.get_wind("82")

weeks = 2
n_input_steps = 24 * 7 * weeks
data = station82_t.Value.values[:n_input_steps]
n_size = data.size
ensembles = dataM.generate_ensambles(data,0.5,100)
cov1 = dataM.calculate_cov(ensembles)
cov2 = dataM.calculate_cov2(ensembles)


#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.linspace(-10,10,n_size)
y = np.linspace(-10,10,n_size)
X, Y = np.meshgrid(x,y)
ax.plot_surface(X, Y, cov2,cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
