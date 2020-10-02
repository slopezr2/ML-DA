#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
from deeplearning import CnnMeteo
from evaluation import Evaluator


window_moving_average=5

#==Assimilation Windows ML_2===
date_DA_ML_2=pd.date_range(start='2019-2-21 20:00:00',end='2019-3-09 18:00:00',freq='H')
date_FC_ML_2=pd.date_range(start='2019-3-9 19:00:00',end='2019-3-12 18:00:00',freq='H')
date_ML_2=date_DA_ML_2.append(date_FC_ML_2)

#==Assimilation Windows ML_3===
date_DA_ML_3=pd.date_range(start='2019-2-21 20:00:00',end='2019-3-10 18:00:00',freq='H')
date_FC_ML_3=pd.date_range(start='2019-3-10 19:00:00',end='2019-3-13 18:00:00',freq='H')
date_ML_3=date_DA_ML_3.append(date_FC_ML_3)

ML_2=Evaluator(date_DA_ML_2,date_FC_ML_2,date_ML_2,ML=2,window_moving_average=5)
ML_3=Evaluator(date_DA_ML_3,date_FC_ML_3,date_ML_3,ML=3,window_moving_average=5)