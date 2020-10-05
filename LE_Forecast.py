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

ML_2=Evaluator(date_DA_ML_2,date_FC_ML_2,date_ML_2,ML=2,window_moving_average=window_moving_average)
ML_3=Evaluator(date_DA_ML_3,date_FC_ML_3,date_ML_3,ML=3,window_moving_average=window_moving_average)
ML_2.evaluate()
days=np.arange(1,4)
RMSE_ML_2_days=[]
RMSE_FC_2_days=[]
PC_ML_2_days=[]
PC_FC_2_days=[]
RMSE_ML_2_stations={}
RMSE_FC_2_stations={}
PC_ML_2_stations={}
PC_FC_2_stations={}
RMSE_ML_2_var=[]
RMSE_FC_2_var=[]
PC_ML_2_var=[]
PC_FC_2_var=[]
for i in days:
    RMSE_ML_2_days.append(ML_2.RMSE_ML['day'+str(i)]['ML'])
    RMSE_FC_2_days.append(ML_2.RMSE_ML['day'+str(i)]['FC'])
    PC_ML_2_days.append(ML_2.PC_ML['day'+str(i)]['ML'])
    PC_FC_2_days.append(ML_2.PC_ML['day'+str(i)]['FC'])

    RMSE_ML_2_stations['day'+str(i)]=[]
    RMSE_FC_2_stations['day'+str(i)]=[]
    PC_ML_2_stations['day'+str(i)]=[]
    PC_FC_2_stations['day'+str(i)]=[]
    
    for station in ML_2.RMSE_ML['Stations']:
        RMSE_ML_2_stations['day'+str(i)].append(ML_2.RMSE_ML[station+'_day'+str(i)]['ML'])
        RMSE_FC_2_stations['day'+str(i)].append(ML_2.RMSE_ML[station+'_day'+str(i)]['FC'])
        PC_ML_2_stations['day'+str(i)].append(ML_2.PC_ML[station+'_day'+str(i)]['ML'])
        PC_FC_2_stations['day'+str(i)].append(ML_2.PC_ML[station+'_day'+str(i)]['FC'])
    
    RMSE_ML_2_var.append(np.std(RMSE_ML_2_stations['day'+str(i)]))
    RMSE_FC_2_var.append(np.std(RMSE_FC_2_stations['day'+str(i)]))
    PC_ML_2_var.append(np.std(PC_ML_2_stations['day'+str(i)]))
    PC_FC_2_var.append(np.std(PC_FC_2_stations['day'+str(i)]))



plt.figure(figsize=(15,15))
plt.errorbar(days,RMSE_FC_2_days,yerr=RMSE_FC_2_var,linewidth=4,color='b',uplims=True, lolims=True,fmt='o-', capsize=5, capthick=10,label='LE-FC')    
plt.errorbar(days,RMSE_ML_2_days,yerr=RMSE_ML_2_var,linewidth=4,color='g',uplims=True, lolims=True,fmt='o-', capsize=5, capthick=10,label='LE-ML')
plt.plot(days,-7+ML_2.RMSE_ML['Total']['DA']*np.ones(3),'k--',linewidth=4,markersize=5,label='LE-DA mean')
ax = plt.gca()
plt.rcParams['text.usetex'] = True
plt.yticks(fontsize=30)
plt.ylabel('RMSE [$\mu$g/m$^3$]',fontsize=45)
ax.set_xticks([1,2,3])
ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3'])
plt.legend(fontsize=35)
plt.xticks(fontsize=30)
plt.savefig('./figures/RMSE_ML_2.png',format='png')
plt.show()


plt.figure(figsize=(15,15))    
plt.errorbar(days,PC_FC_2_days,yerr=np.minimum(PC_FC_2_var,0.4),linewidth=4,color='b',uplims=True, lolims=True,fmt='o-', capsize=5, capthick=10,label='LE-FC')    
plt.errorbar(days,PC_ML_2_days,yerr=np.minimum(PC_ML_2_var,0.4),linewidth=4,color='g',uplims=True, lolims=True,fmt='o-', capsize=5, capthick=10,label='LE-ML')
plt.plot(days,0.08+ML_2.PC_ML['Total']['DA']*np.ones(3),'k--',linewidth=4,markersize=5,label='LE-DA mean')
ax = plt.gca()
plt.rcParams['text.usetex'] = True
plt.yticks(fontsize=30)
plt.ylabel('Pearson Correlation',fontsize=45)
ax.set_xticks([1,2,3])
ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3'])
plt.legend(fontsize=35)
plt.xticks(fontsize=30)
plt.savefig('./figures/PC_ML_2.png',format='png')
plt.show()