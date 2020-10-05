#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation import Evaluator
import shapefile as shp
import seaborn as sns


window_moving_average=5


lat_stations=[6.2525611,6.2633696,6.1856666,6.3790379,6.0990806,6.1684971,6.1825418,6.1523128,6.0930777,
              6.1555305,6.2218938,6.2589092,6.4369602,6.3453598,6.1998701,6.2778502,6.2904806,6.3375502,
              6.1686831,6.1433334,6.236361]

lon_stations=[-75.5695801,-75.5770035,-75.5972061,-75.4509125,-75.6386261,-75.6443558,-75.5506363,-75.6274872,
              -75.637764,-75.6441727,-75.6106033,-75.5482635,-75.3303986,-75.5047531,-75.5609512,-75.6364288,
              -75.5555191,-75.5678024,-75.5819702,-75.6202774,-75.4984741]

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

sf = shp.Reader('/home/slopezr2/Documents/Municipios_AreaMetropolitana.shp')

sns.set(style='whitegrid', palette='pastel', color_codes=True)
sns.mpl.rc('figure', figsize=(10,6))



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
    
    
plt.figure(figsize = (10,6))
plt.rcParams['text.usetex'] = True
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x, y, 'k')
    ax = plt.gca()
    ax.fill(x,y, color=[0.8,0.8,0.8],zorder=0)  
SC=plt.scatter(x=lon_stations,y=lat_stations,c=RMSE_ML_2_mean_days,zorder=10,cmap='Reds') 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
cbar=plt.colorbar(SC)    
cbar.ax.tick_params(labelsize=14)
plt.savefig('./figures/Map_RMSE_ML_2.png',format='png')
plt.show()
    
plt.figure(figsize = (10,6))
plt.rcParams['text.usetex'] = True
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x, y, 'k')
    ax = plt.gca()
    ax.fill(x,y, color=[0.8,0.8,0.8],zorder=0)  
SC=plt.scatter(x=lon_stations,y=lat_stations,c=RMSE_FC_2_mean_days,zorder=10,cmap='Reds') 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
cbar=plt.colorbar(SC)    
cbar.ax.tick_params(labelsize=14)
plt.savefig('./figures/Map_RMSE_FC_2.png',format='png')
plt.show()
    
plt.figure(figsize = (10,6))
plt.rcParams['text.usetex'] = True
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x, y, 'k')
    ax = plt.gca()
    ax.fill(x,y, color=[0.8,0.8,0.8],zorder=0)  
SC=plt.scatter(x=lon_stations,y=lat_stations,c=PC_ML_2_mean_days,zorder=10,cmap='Reds') 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
cbar=plt.colorbar(SC)    
plt.clim(-0.2,1)
cbar.ax.tick_params(labelsize=14)
plt.savefig('./figures/Map_PC_ML_2.png',format='png')
plt.show()
    
plt.figure(figsize = (10,6))
plt.rcParams['text.usetex'] = True
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x, y, 'k')
    ax = plt.gca()
    ax.fill(x,y, color=[0.8,0.8,0.8],zorder=0)  
SC=plt.scatter(x=lon_stations,y=lat_stations,c=PC_FC_2_mean_days,zorder=10,cmap='Reds') 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
cbar=plt.colorbar(SC) 
plt.clim(-0.2,1)   
cbar.ax.tick_params(labelsize=14)
plt.savefig('./figures/Map_PC_FC_2.png',format='png')
plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    