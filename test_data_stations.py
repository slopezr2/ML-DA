import numpy as np
from datamanager import DataManager
from datamanager import Combiner
import matplotlib.pyplot as plt
import pandas as pd

dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])
stations=pd.read_csv('./data/Lista_estaciones.csv',delimiter=',')
stations_SIATA=np.array(stations['PM25'].values).astype('str')
stations_Meteo=np.array(stations['Meteo'].values).astype('str')

#graph_type. If it is complete, Graph all data
#If it is period, graph data for simulation period

graph_type="period"



n_input_steps = 24 * 7 * 2
n_output_steps=24*3
pre_processor=Combiner()
n_train = 9500
for station in range(len(stations_SIATA)):
    station_pm25 = dataM.get_pm25(stations_SIATA[station])
    station_t = dataM.get_temperature(stations_Meteo[station])
    station_w = dataM.get_wind(stations_Meteo[station])
    number_samples=min(len(station_t.Value.values),len(station_pm25.CONCENTRATION.values))
    station_pm25 =station_pm25[0:number_samples]
    station_t =station_t[0:number_samples]
    station_w =station_w[0:number_samples]
    print("=============================================================")
    print("SIATA Station " + stations_SIATA[station]+" Shape "+str(station_pm25.shape))
    print("Temperature Station " + stations_Meteo[station]+" Shape "+str(station_t.shape))
    print("Temperature Station " + stations_Meteo[station]+" Shape "+str(station_w.shape))
    #print("Initial and final date SIATA "+str(dates_pm25[1])+" -- "+str(dates_pm25[-1]))
    #print("Initial and final date Meteo "+str(dates_t[1])+" -- "+str(dates_t[-1]))
    if graph_type=="complete":
        dates_pm25=(station_pm25.Date.values)
        dates_t=(station_t.Date.values)
        
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
    if graph_type=="period":
        X, y = pre_processor.combine(n_input_steps, n_output_steps, station_t.Value.values, station_w.Value.values,
                                     station_pm25.CONCENTRATION.values,station_t.Date.dt.dayofweek.values,station_t.Date.dt.hour.values , station_pm25.CONCENTRATION.values)
        dates=(station_pm25.Date.values)
        dates=np.transpose(dates)
        datesx,datesy=pre_processor.create_dataset(dates,look_back=n_input_steps,step_forecast=n_output_steps)
        
        scalerC = pre_processor.scalers[0]
        temperature=X[n_train, :, 0]
        temperature = temperature.reshape((1, temperature.shape[0]))
        temperature=scalerC.inverse_transform(temperature)
        
        scalerC = pre_processor.scalers[1]
        wind=X[n_train, :, 1]
        wind = wind.reshape((1, wind.shape[0]))
        wind=scalerC.inverse_transform(wind)
        
        scalerC = pre_processor.scalers[2]
        pm25=X[n_train, :, 2]
        pm25 = pm25.reshape((1, pm25.shape[0]))
        pm25=scalerC.inverse_transform(pm25)
        
        plt.figure(dpi=1200)
        plt.plot(datesx[n_train,:],temperature[0,:])
        ax = plt.gca()
        just_day=lambda t: t[6:10]
        just_days=np.array([just_day(t) for t in datesx[n_train,:]])
        custom_ticks = np.arange(5, 336, 48)
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(just_days[custom_ticks])
        plt.title("Temperature Station " + stations_Meteo[station]+" Shape "+str(station_t.shape))
        plt.show()
        plt.figure(dpi=1200)
        plt.plot(datesx[n_train,:],wind[0,:])
        ax = plt.gca()
        just_day=lambda t: t[6:10]
        just_days=np.array([just_day(t) for t in datesx[n_train,:]])
        custom_ticks = np.arange(5, 336, 48)
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(just_days[custom_ticks])
        plt.title("Wind Station " + stations_Meteo[station]+" Shape "+str(station_t.shape))
        plt.show()
        plt.figure(dpi=1200)
        plt.plot(datesx[n_train,:],pm25[0,:])
        ax = plt.gca()
        just_day=lambda t: t[6:10]
        just_days=np.array([just_day(t) for t in datesx[n_train,:]])
        custom_ticks = np.arange(5, 336, 48)
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(just_days[custom_ticks])
        plt.title("PM25 Station " + stations_SIATA[station]+" Shape "+str(station_pm25.shape))
        plt.show()