#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

window_moving_average=2
save=True

tpm25_ML_2_file=netCDF4.Dataset('./data/tpm25_ML_2.nc','r')
tpm25_FC_2_file=netCDF4.Dataset('./data/tpm25_FC_2.nc','r')
Y_tpm25_ML_2_file=netCDF4.Dataset('./data/Y_tpm25_ML_2.nc','r')

tpm25_ML_3_file=netCDF4.Dataset('./data/tpm25_ML_3.nc','r')
tpm25_FC_3_file=netCDF4.Dataset('./data/tpm25_FC_3.nc','r')
Y_tpm25_ML_3_file=netCDF4.Dataset('./data/Y_tpm25_ML_3.nc','r')

Template_file=netCDF4.Dataset('./data/Template_validation_file.nc','r')

Xa_PM25_ML_2=tpm25_ML_2_file.variables['tpm25'][:]
Xa_PM25_ML_2=np.array(Xa_PM25_ML_2*1e9)
Xa_PM25_FC_2=tpm25_FC_2_file.variables['tpm25'][:]
Xa_PM25_FC_2=np.array(Xa_PM25_FC_2*1e9)

Y_PM25_ML_2=Y_tpm25_ML_2_file.variables['tpm25_y'][:]
Y_PM25_ML_2=np.array(Y_PM25_ML_2*1e9)
Y_PM25_ML_2[np.where(Y_PM25_ML_2>200)]=np.NaN
Y_PM25_ML_2[np.where(Y_PM25_ML_2<0)]=np.NaN

Xa_PM25_ML_3=tpm25_ML_3_file.variables['tpm25'][:]
Xa_PM25_ML_3=np.array(Xa_PM25_ML_3*1e9)
Xa_PM25_FC_3=tpm25_FC_3_file.variables['tpm25'][:]
Xa_PM25_FC_3=np.array(Xa_PM25_FC_3*1e9)

Y_PM25_ML_3=Y_tpm25_ML_3_file.variables['tpm25_y'][:]
Y_PM25_ML_3=np.array(Y_PM25_ML_3*1e9)
Y_PM25_ML_3[np.where(Y_PM25_ML_3>200)]=np.NaN
Y_PM25_ML_3[np.where(Y_PM25_ML_3<0)]=np.NaN
names_stations=Template_file.variables['station_name'][:]

#==Assimilation Windows ML_2===
date_DA_ML_2=pd.date_range(start='2019-2-21 20:00:00',end='2019-3-09 18:00:00',freq='H')
date_FC_ML_2=pd.date_range(start='2019-3-9 19:00:00',end='2019-3-12 18:00:00',freq='H')
date_ML_2=date_DA_ML_2.append(date_FC_ML_2)

#==Assimilation Windows ML_3===
date_DA_ML_3=pd.date_range(start='2019-2-21 20:00:00',end='2019-3-10 18:00:00',freq='H')
date_FC_ML_3=pd.date_range(start='2019-3-10 19:00:00',end='2019-3-13 18:00:00',freq='H')
date_ML_3=date_DA_ML_3.append(date_FC_ML_3)
for i in range(len(names_stations)):
    if np.sum(np.isnan(Y_PM25_ML_2[:,0,i]))>100:
        continue
    name_station=str(names_stations[i,(np.logical_not(names_stations[i].mask))])
    name_station=name_station.replace("]",'')
    name_station=name_station.replace("[",'')
    name_station=name_station.replace("b",'')
    name_station=name_station.replace("'",'')
    name_station=name_station.replace(" ",'')
    plt.figure(figsize=(30,15))
    plt.title(name_station,fontsize=30)
    plt.plot(date_ML_2[48:],pd.Series(Y_PM25_ML_2[48:,0,i]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values,'r*-',linewidth=3,markersize=10,label='Real Data')
    plt.plot(date_DA_ML_2[48:],pd.Series(Xa_PM25_ML_2[48:len(date_DA_ML_2),0,i]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values,'k',linewidth=4,markersize=10,label='LE-DA')
    plt.plot(date_FC_ML_2[:-1],pd.Series(Xa_PM25_FC_2[:,0,i]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values,'b',linewidth=4,markersize=10,label='LE-FC')
    plt.plot(date_FC_ML_2,pd.Series(Xa_PM25_ML_2[len(date_DA_ML_2):,0,i]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values,'g',linewidth=4,markersize=10,label='LE-ML')
    plt.axvline(date_FC_ML_2[0],linewidth=3,linestyle='--',color=[0.3,0.3,0.3])
    ax = plt.gca()
    plt.rcParams['text.usetex'] = True
    plt.yticks(fontsize=30)
    plt.ylabel('PM$_{2.5}$ Concentration [$\mu$g/m$^3$]',fontsize=45)
    plt.grid(axis='x')
    plt.legend(fontsize=35)
    plt.xticks(fontsize=30)
    ax.set_xlim(date_ML_2[48], date_ML_2[-1])
    ax.set_ylim(0, 150)
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    if save :
            plt.savefig("./figures/"+name_station+"_ML_2.png",format="png")
    plt.show()
    plt.figure(figsize=(30,15))
    plt.title(name_station,fontsize=30)
    plt.plot(date_ML_3[72:],pd.Series(Y_PM25_ML_3[72:,0,i]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values,'r*-',linewidth=3,markersize=10,label='Real Data')
    plt.plot(date_DA_ML_3[72:],pd.Series(Xa_PM25_ML_3[72:len(date_DA_ML_3),0,i]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values,'k',linewidth=4,markersize=10,label='LE-DA')
    plt.plot(date_FC_ML_3[:-1],pd.Series(Xa_PM25_FC_3[:,0,i]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values,'b',linewidth=4,markersize=10,label='LE-FC')
    plt.plot(date_FC_ML_3,pd.Series(Xa_PM25_ML_3[len(date_DA_ML_3):,0,i]).rolling(window=window_moving_average,min_periods=1,center=False).mean().values,'g',linewidth=4,markersize=10,label='LE-ML')
    plt.axvline(date_FC_ML_3[0],linewidth=3,linestyle='--',color=[0.3,0.3,0.3])
    ax = plt.gca()
    plt.rcParams['text.usetex'] = True
    plt.yticks(fontsize=30)
    plt.ylabel('PM$_{2.5}$ Concentration [$\mu$g/m$^3$]',fontsize=45)
    plt.grid(axis='x')
    plt.legend(fontsize=35)
    plt.xticks(fontsize=30)
    ax.set_xlim(date_ML_3[72], date_ML_3[-1])
    ax.set_ylim(0, 150)
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    if save :
            plt.savefig("./figures/"+name_station+"_ML_3.png",format="png")
    plt.show()
