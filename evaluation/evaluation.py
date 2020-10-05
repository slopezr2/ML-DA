import numpy as np
import pandas as pd
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import netCDF4
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
import matplotlib.dates as mdates

class Evaluator:
    def __init__(self,date_DA_ML,date_FC_ML,date_ML,window_moving_average=5,ML=2):
        self.window_moving_average=window_moving_average
        self.date_DA_ML=date_DA_ML
        self.date_FC_ML=date_FC_ML
        self.date_ML=date_ML
        self.date_DA_Y=pd.date_range(start='2019-2-21 20:00:00',end='2019-3-10 18:00:00',freq='H')
        self.date_FC_Y=pd.date_range(start='2019-3-10 19:00:00',end='2019-3-15 18:00:00',freq='H')
        self.date_Y=self.date_DA_Y.append(self.date_FC_Y)
        

        
        self.ML=ML
        ML=str(ML)
        
        self.tpm25_ML_file=netCDF4.Dataset('./data/tpm25_ML_'+ML+'.nc','r')
        self.tpm25_FC_file=netCDF4.Dataset('./data/tpm25_FC_'+ML+'.nc','r')
        self.Y_tpm25_ML_file=netCDF4.Dataset('./data/Y_tpm25_ML_'+ML+'.nc','r')
        self.xb_tpm25_ML_file=netCDF4.Dataset('./data/xb_tpm25.nc','r')
        self.Y_tpm25_file=netCDF4.Dataset('./data/Y_tpm25.nc','r')
        self.Template_file=netCDF4.Dataset('./data/Template_validation_file.nc','r')
        
        self.Xa_PM25_ML=self.tpm25_ML_file.variables['tpm25'][:]
        self.Xa_PM25_ML=np.array(self.Xa_PM25_ML*1e9)
        
        self.Xa_PM25_FC=self.tpm25_FC_file.variables['tpm25'][:]
        self.Xa_PM25_FC=np.array(self.Xa_PM25_FC*1e9)
        
        self.Y_PM25_ML=self.Y_tpm25_ML_file.variables['tpm25_y'][:]
        self.Y_PM25_ML=np.array(self.Y_PM25_ML*1e9)
        self.Y_PM25_ML[np.where(self.Y_PM25_ML>200)]=np.NaN
        self.Y_PM25_ML[np.where(self.Y_PM25_ML<0)]=np.NaN
        
        self.Xb_PM25_ML=self.xb_tpm25_ML_file.variables['tpm25'][:]
        self.Xb_PM25_ML=np.array(self.Xb_PM25_ML*1e9)
        
        self.Y_PM25=self.Y_tpm25_file.variables['tpm25_y'][:]
        self.Y_PM25=np.array(self.Y_PM25*1e9)
        self.Y_PM25[np.where(self.Y_PM25>200)]=np.NaN
        self.Y_PM25[np.where(self.Y_PM25<0)]=np.NaN
        
        self.names_stations=self.Template_file.variables['station_name'][:]

    def evaluate(self):
        self.RMSE_ML={}
        self.PC_ML={}
        self.RMSE_ML['Stations']=[]
        self.PC_ML['Stations']=[]
        evaluated_stations=0
        RMSE_ML_sum_DA=[]
        PC_ML_sum_DA=[]
        RMSE_ML_sum_total=[]
        RMSE_FC_sum_total=[]
        PC_ML_sum_total=[]
        PC_FC_sum_total=[]
        RMSE_ML_sum_day1=[]
        RMSE_FC_sum_day1=[]
        PC_ML_sum_day1=[]
        PC_FC_sum_day1=[]
        RMSE_ML_sum_day2=[]
        RMSE_FC_sum_day2=[]
        PC_ML_sum_day2=[]
        PC_FC_sum_day2=[]
        RMSE_ML_sum_day3=[]
        RMSE_FC_sum_day3=[]
        PC_ML_sum_day3=[]
        PC_FC_sum_day3=[]
        for i in range(len(self.names_stations)):
            if np.sum(np.isnan(self.Y_PM25[:,0,i]))>100:
                continue
            evaluated_stations=evaluated_stations+1
            Y_real=pd.Series(self.Y_PM25[self.date_Y.searchsorted(self.date_FC_ML[0]):self.date_Y.searchsorted(self.date_FC_ML[-1]),0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values
            nan_index=np.logical_not(np.isnan(Y_real))
            Y_ML=pd.Series(self.Xa_PM25_ML[len(self.date_DA_ML):-1,0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values
            Y_FC=pd.Series(self.Xa_PM25_FC[:,0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values
            name_station=str(self.names_stations[i,(np.logical_not(self.names_stations[i].mask))])
            name_station=name_station.replace("]",'')
            name_station=name_station.replace("[",'')
            name_station=name_station.replace("b",'')
            name_station=name_station.replace("'",'')
            name_station=name_station.replace(" ",'')
            self.RMSE_ML['Stations'].append(name_station)
            self.PC_ML['Stations'].append(name_station)
            self.RMSE_ML[name_station+'_total']={}
            self.RMSE_ML[name_station+'_total']['ML']=sqrt(mean_squared_error(Y_real[nan_index],Y_ML[nan_index]))
            self.RMSE_ML[name_station+'_total']['FC']=sqrt(mean_squared_error(Y_real[nan_index],Y_FC[nan_index]))
            self.PC_ML[name_station+'_total']={}
            self.PC_ML[name_station+'_total']['ML']=pearsonr(Y_real[nan_index],Y_ML[nan_index])[0]
            self.PC_ML[name_station+'_total']['FC']=pearsonr(Y_real[nan_index],Y_FC[nan_index])[0]
            self.RMSE_ML[name_station+'_day1']={}
            self.RMSE_ML[name_station+'_day1']['ML']=sqrt(mean_squared_error(Y_real[nan_index][0:24],Y_ML[nan_index][0:24]))
            self.RMSE_ML[name_station+'_day1']['FC']=sqrt(mean_squared_error(Y_real[nan_index][0:24],Y_FC[nan_index][0:24]))
            self.PC_ML[name_station+'_day1']={}
            self.PC_ML[name_station+'_day1']['ML']=pearsonr(Y_real[nan_index][0:24],Y_ML[nan_index][0:24])[0]
            self.PC_ML[name_station+'_day1']['FC']=pearsonr(Y_real[nan_index][0:24],Y_FC[nan_index][0:24])[0]
            self.RMSE_ML[name_station+'_day2']={}
            self.RMSE_ML[name_station+'_day2']['ML']=sqrt(mean_squared_error(Y_real[nan_index][24:48],Y_ML[nan_index][24:48]))
            self.RMSE_ML[name_station+'_day2']['FC']=sqrt(mean_squared_error(Y_real[nan_index][24:48],Y_FC[nan_index][24:48]))
            self.PC_ML[name_station+'_day2']={}
            self.PC_ML[name_station+'_day2']['ML']=pearsonr(Y_real[nan_index][24:48],Y_ML[nan_index][24:48])[0]
            self.PC_ML[name_station+'_day2']['FC']=pearsonr(Y_real[nan_index][24:48],Y_FC[nan_index][24:48])[0]
            self.RMSE_ML[name_station+'_day3']={}
            self.RMSE_ML[name_station+'_day3']['ML']=sqrt(mean_squared_error(Y_real[nan_index][48:],Y_ML[nan_index][48:]))
            self.RMSE_ML[name_station+'_day3']['FC']=sqrt(mean_squared_error(Y_real[nan_index][48:],Y_FC[nan_index][48:]))
            self.PC_ML[name_station+'_day3']={}
            self.PC_ML[name_station+'_day3']['ML']=pearsonr(Y_real[nan_index][48:],Y_ML[nan_index][48:])[0]
            self.PC_ML[name_station+'_day3']['FC']=pearsonr(Y_real[nan_index][48:],Y_FC[nan_index][48:])[0]
            #==Evaluation in the Assimilation windows===
            Y_real=pd.Series(self.Y_PM25[self.date_Y.searchsorted(self.date_DA_ML[0]):self.date_Y.searchsorted(self.date_DA_ML[-1]),0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values
            nan_index=np.logical_not(np.isnan(Y_real))
            Y_DA=pd.Series(self.Xa_PM25_ML[0:len(self.date_DA_ML)-1,0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values
            self.RMSE_ML[name_station+'_total']['DA']=sqrt(mean_squared_error(Y_real[nan_index][-72:-1],Y_DA[nan_index][-72:-1]))
            self.PC_ML[name_station+'_total']['DA']=pearsonr(Y_real[nan_index][-72:-1],Y_DA[nan_index][-72:-1])[0]
            #===Sum for total across the stations====
            RMSE_ML_sum_total.append(self.RMSE_ML[name_station+'_total']['ML'])
            RMSE_FC_sum_total.append(self.RMSE_ML[name_station+'_total']['FC'])
            PC_ML_sum_total.append(self.PC_ML[name_station+'_total']['ML'])
            PC_FC_sum_total.append(self.PC_ML[name_station+'_total']['FC'])
            RMSE_ML_sum_day1.append(self.RMSE_ML[name_station+'_day1']['ML'])
            RMSE_FC_sum_day1.append(self.RMSE_ML[name_station+'_day1']['FC'])
            PC_ML_sum_day1.append(self.PC_ML[name_station+'_day1']['ML'])
            PC_FC_sum_day1.append(self.PC_ML[name_station+'_day1']['FC'])
            RMSE_ML_sum_day2.append(self.RMSE_ML[name_station+'_day2']['ML'])
            RMSE_FC_sum_day2.append(self.RMSE_ML[name_station+'_day2']['FC'])
            PC_ML_sum_day2.append(self.PC_ML[name_station+'_day2']['ML'])
            PC_FC_sum_day2.append(self.PC_ML[name_station+'_day2']['FC'])
            RMSE_ML_sum_day3.append(self.RMSE_ML[name_station+'_day3']['ML'])
            RMSE_FC_sum_day3.append(self.RMSE_ML[name_station+'_day3']['FC'])
            PC_ML_sum_day3.append(self.PC_ML[name_station+'_day3']['ML'])
            PC_FC_sum_day3.append(self.PC_ML[name_station+'_day3']['FC'])
            RMSE_ML_sum_DA.append(self.RMSE_ML[name_station+'_total']['DA'])
            PC_ML_sum_DA.append(self.PC_ML[name_station+'_total']['DA'])
            
        self.RMSE_ML['Total']={}
        self.RMSE_ML['Total']['ML']=np.median(RMSE_ML_sum_total)
        self.RMSE_ML['Total']['FC']=np.median(RMSE_FC_sum_total)
        self.RMSE_ML['Total']['DA']=np.median(RMSE_ML_sum_DA)
        self.PC_ML['Total']={}
        self.PC_ML['Total']['ML']=np.median(PC_ML_sum_total)
        self.PC_ML['Total']['FC']=np.median(PC_FC_sum_total)
        self.PC_ML['Total']['DA']=np.median(PC_ML_sum_DA)
        self.RMSE_ML['day1']={}
        self.RMSE_ML['day1']['ML']=np.median(RMSE_ML_sum_day1)
        self.RMSE_ML['day1']['FC']=np.median(RMSE_FC_sum_day1)
        self.PC_ML['day1']={}
        self.PC_ML['day1']['ML']=np.median(PC_ML_sum_day1)
        self.PC_ML['day1']['FC']=np.median(PC_FC_sum_day1)
        self.RMSE_ML['day2']={}
        self.RMSE_ML['day2']['ML']=np.median(RMSE_ML_sum_day2)
        self.RMSE_ML['day2']['FC']=np.median(RMSE_FC_sum_day2)
        self.PC_ML['day2']={}
        self.PC_ML['day2']['ML']=np.median(PC_ML_sum_day2)
        self.PC_ML['day2']['FC']=np.median(PC_FC_sum_day2)
        self.RMSE_ML['day3']={}
        self.RMSE_ML['day3']['ML']=np.median(RMSE_ML_sum_day3)
        self.RMSE_ML['day3']['FC']=np.median(RMSE_FC_sum_day3)
        self.PC_ML['day3']={}
        self.PC_ML['day3']['ML']=np.median(PC_ML_sum_day3)
        self.PC_ML['day3']['FC']=np.median(PC_FC_sum_day3)
    
    def graph(self,save=False):
        for i in range(len(self.names_stations)):
            if np.sum(np.isnan(self.Y_PM25[:,0,i]))>100:
                continue
            name_station=str(self.names_stations[i,(np.logical_not(self.names_stations[i].mask))])
            name_station=name_station.replace("]",'')
            name_station=name_station.replace("[",'')
            name_station=name_station.replace("b",'')
            name_station=name_station.replace("'",'')
            name_station=name_station.replace(" ",'')
            plt.figure(figsize=(30,15))
            plt.title(name_station,fontsize=30)
            plt.plot(self.date_Y[24*self.ML:],pd.Series(self.Y_PM25[24*self.ML:,0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values,'r*-',linewidth=3,markersize=10,label='Real Data')
            plt.plot(self.date_DA_ML[24*self.ML:],pd.Series(self.Xa_PM25_ML[24*self.ML:len(self.date_DA_ML),0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values,'k',linewidth=4,markersize=10,label='LE-DA')
            plt.plot(self.date_FC_ML[:-1],pd.Series(self.Xa_PM25_FC[:,0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values,'b',linewidth=4,markersize=10,label='LE-FC')
            plt.plot(self.date_FC_ML,pd.Series(self.Xa_PM25_ML[len(self.date_DA_ML):,0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values,'g',linewidth=4,markersize=10,label='LE-ML')
            #plt.plot(self.date_Y[24*self.ML:],pd.Series(self.Xb_PM25_ML[24*self.ML:,0,i]).rolling(window=self.window_moving_average,min_periods=1,center=False).mean().values,'k--',linewidth=3,markersize=10,label='LE')
            plt.axvline(self.date_FC_ML[0],linewidth=3,linestyle='--',color=[0.3,0.3,0.3])
            ax = plt.gca()
            plt.rcParams['text.usetex'] = True
            plt.yticks(fontsize=30)
            plt.ylabel('PM$_{2.5}$ Concentration [$\mu$g/m$^3$]',fontsize=45)
            plt.grid(axis='x')
            plt.legend(fontsize=35)
            plt.xticks(fontsize=30)
            ax.set_xlim(self.date_ML[24*self.ML], self.date_ML[-1])
            ax.set_ylim(0, 150)
            ax.xaxis.set_major_locator(plt.MaxNLocator(20))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            if save :
                    plt.savefig('./figures/'+name_station+'_ML_'+str(self.ML)+'.png',format='png')
            plt.show()




