#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation import Evaluator



window_moving_average=5

RMSE_LSTM_2=[np.NaN,np.NaN,11.00,10.22,11.04,18.16,12.86,8.49,10.41,7.47,6.03,9.47,6.75,5.74,13.18,7.54,8.42,7.78,9.53,np.NaN,np.NaN]


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
RMSE_ML_2_stations={}
RMSE_FC_2_stations={}
PC_ML_2_stations={}
PC_FC_2_stations={}
for i in days:
    RMSE_ML_2_stations['day'+str(i)]=[]
    RMSE_FC_2_stations['day'+str(i)]=[]
    PC_ML_2_stations['day'+str(i)]=[]
    PC_FC_2_stations['day'+str(i)]=[]
    for station in ML_2.RMSE_ML['Stations']:
        RMSE_ML_2_stations['day'+str(i)].append(ML_2.RMSE_ML[station+'_day'+str(i)]['ML'])
        RMSE_FC_2_stations['day'+str(i)].append(ML_2.RMSE_ML[station+'_day'+str(i)]['FC'])
        PC_ML_2_stations['day'+str(i)].append(ML_2.PC_ML[station+'_day'+str(i)]['ML'])
        PC_FC_2_stations['day'+str(i)].append(ML_2.PC_ML[station+'_day'+str(i)]['FC'])

RMSE_ML_2_mean_days=[] 
RMSE_FC_2_mean_days=[]
PC_ML_2_mean_days=[] 
PC_FC_2_mean_days=[] 
for station in range(len(ML_2.RMSE_ML['Stations'])):
    RMSE_ML_2_mean_days.append(np.median([RMSE_ML_2_stations['day1'][station],RMSE_ML_2_stations['day2'][station],RMSE_ML_2_stations['day3'][station]]))
    RMSE_FC_2_mean_days.append(np.median([RMSE_FC_2_stations['day1'][station],RMSE_FC_2_stations['day2'][station],RMSE_FC_2_stations['day3'][station]]))
    PC_ML_2_mean_days.append(np.median([PC_ML_2_stations['day1'][station],PC_ML_2_stations['day2'][station],PC_ML_2_stations['day3'][station]]))
    PC_FC_2_mean_days.append(np.median([PC_FC_2_stations['day1'][station],PC_FC_2_stations['day2'][station],PC_FC_2_stations['day3'][station]]))



    
    
plt.figure(figsize=(40,20))
ax = plt.gca()
plt.rcParams['text.usetex'] = True
plt.yticks(fontsize=30)
plt.ylabel('RMSE [$\mu$g/m$^3$]',fontsize=45)
plt.xlabel('Stations',fontsize=45)
plt.grid(axis='y')
x = np.arange(len(ML_2.RMSE_ML['Stations'])) 
width = 0.25
ax.bar(x - width, RMSE_FC_2_mean_days, width,color='b', label='LE-FC')
ax.bar(x, RMSE_ML_2_mean_days, width,color='g', label='LE-ML')
ax.bar(x + width, RMSE_LSTM_2, width,color='r', label='ML')
ax.set_xticks(range(21))
ax.set_xticklabels([ML_2.RMSE_ML['Stations'][i][7:] for i in range(21)])
plt.legend(fontsize=35)
plt.xticks(fontsize=30)

plt.savefig('./figures/Bar_RMSE_ML_2.png',format='png')
plt.show()
    
    
    