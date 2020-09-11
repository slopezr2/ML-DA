import numpy as np
from datamanager import DataManager
import matplotlib.pyplot as plt
import pandas as pd

dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])
stations=pd.read_csv('./data/Lista_estaciones.csv',delimiter=',')
stations_SIATA=np.array(stations['PM25'].values).astype('str')
stations_Meteo=np.array(stations['Meteo'].values).astype('str')

for station in range(len(stations_SIATA)):
    station_pm25 = dataM.get_pm25(stations_SIATA[station])
    station_t = dataM.get_temperature(stations_Meteo[station])
    station_w = dataM.get_wind(stations_Meteo[station])
    number_samples=min(len(station_t.Value.values),len(station_pm25.CONCENTRATION.values))
    station_pm25 =station_pm25[0:number_samples]
    station_t =station_t[0:number_samples]
    station_w =station_w[0:number_samples]
    dates_pm25=(station_pm25.Date.values)
    dates_t=(station_t.Date.values)
    print("=============================================================")
    print("SIATA Station " + stations_SIATA[station]+" Shape "+str(station_pm25.shape))
    print("Temperature Station " + stations_Meteo[station]+" Shape "+str(station_t.shape))
    print("Temperature Station " + stations_Meteo[station]+" Shape "+str(station_w.shape))
    #print("Initial and final date SIATA "+str(dates_pm25[1])+" -- "+str(dates_pm25[-1]))
    #print("Initial and final date Meteo "+str(dates_t[1])+" -- "+str(dates_t[-1]))
    plt.plot(station_t.Value.values)
    plt.title("Temperature Station " + stations_Meteo[station]+" Shape "+str(station_t.shape))
    plt.show()
    plt.Figure()
    plt.plot(station_w.Value.values)
    plt.title("Wind Station " + stations_Meteo[station]+" Shape "+str(station_t.shape))
    plt.show()
    plt.Figure()
    plt.plot(station_pm25.CONCENTRATION.values)
    plt.title("PM25 Station " + stations_SIATA[station]+" Shape "+str(station_pm25.shape))
    plt.show()