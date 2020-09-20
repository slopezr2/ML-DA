from datamanager import DataManager, Combiner, CsvWriter
from deeplearning import CnnMeteo
import os
import pandas as pd
from datetime import datetime
import warnings
import numpy as np
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Parameters to select the LSTM architecture 
semana = 2
n_LSTM_hidden_layers = 2
n_cells = 200
n_features = 5
# Value to graph a moving average of observations. Just for graphical purposes
window_moving_average = 5
# Correction of time zone (LOTOS-EUROS time zone is UTC-0, Colombia is UTC-5)
UTC=5


days_forecast = 1
dataM = DataManager(path="data/", filter_items=["pm25", "temperature", "wind"])

n_input_steps = 24 * 7 * semana
n_output_steps = 24 * 3

stations = pd.read_csv('./data/Lista_estaciones.csv', delimiter=',')
stations_SIATA = np.array(stations['PM25'].values).astype('str')
stations_Meteo = np.array(stations['Meteo'].values).astype('str')

mls = CnnMeteo(n_input_steps, n_features, n_output_steps, drop=True, n_LSTM_hidden_layers=n_LSTM_hidden_layers,
               n_cells=n_cells)
csv_writer_1 = CsvWriter()
csv_writer_2 = CsvWriter()
csv_writer_3 = CsvWriter()



for j in range(days_forecast):
    for station in range(len(stations_SIATA)):
    
        print(stations_SIATA[station])
        station_pm25 = dataM.get_pm25(stations_SIATA[station])
        station_t = dataM.get_temperature(stations_Meteo[station])
        station_w = dataM.get_wind(stations_Meteo[station])
        number_samples = min(len(station_t.Value.values), len(station_pm25.CONCENTRATION.values))
        station_pm25 = station_pm25[0:number_samples]
        station_t = station_t[0:number_samples]
        station_w = station_w[0:number_samples]
        pre_processor = Combiner()
        X, y = pre_processor.combine(n_input_steps, n_output_steps, station_t.Value.values, station_w.Value.values,
                                     station_pm25.CONCENTRATION.values, station_t.Date.dt.dayofweek.values,
                                     station_t.Date.dt.hour.values, station_pm25.CONCENTRATION.values)
    
        # Create Model
        n_train = 9500+527
    
    
        n_features = X.shape[2]
    
        dates = (station_pm25.Date.values)
        dates = np.transpose(dates)
        datesx, datesy = pre_processor.create_dataset(dates, look_back=n_input_steps, step_forecast=n_output_steps)
        file_name = "./models/Station" + stations_SIATA[station] + ".h5"
        mls.load(file_name)
        
        scalerC = pre_processor.scalers[2]
        # demonstrate prediction
        x_input = X[n_train + 1 + j * 24, :, :]
        x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
        yhat = mls.predict(x_input, verbose=1)
        print("predicted")
        scalerC = pre_processor.scalers[5]
        yhat = scalerC.inverse_transform(yhat)
        station_str = str(stations_SIATA[station])
        ns = 1e-9
        date_ini_1 = datetime.strptime(datesy[n_train+UTC,1],'%Y-%m-%d %H:%M:%S')
        date_ini_2 = datetime.strptime(datesy[n_train+UTC,1+24],'%Y-%m-%d %H:%M:%S')
        date_ini_3 = datetime.strptime(datesy[n_train+UTC,1+48],'%Y-%m-%d %H:%M:%S')
        hours = 1
        csv_writer_1.add(yhat[0,0:24],'Station'+station_str,'pm5','ug/m3',1,date_ini_1,hours)
        csv_writer_2.add(yhat[0,24:48],'Station'+station_str,'pm5','ug/m3',1,date_ini_2,hours)
        csv_writer_3.add(yhat[0,48:72],'Station'+station_str,'pm5','ug/m3',1,date_ini_3,hours)

    csv_writer_1.write_observation('observations/output/Observaciones_SIATA_tpm25_2019'+datesy[n_train+UTC,1][5:7]+datesy[n_train,1][8:10]+'.csv')
    csv_writer_2.write_observation('observations/output/Observaciones_SIATA_tpm25_2019'+datesy[n_train+UTC,1+24][5:7]+datesy[n_train,1+24][8:10]+'.csv')
    csv_writer_3.write_observation('observations/output/Observaciones_SIATA_tpm25_2019'+datesy[n_train+UTC,1+48][5:7]+datesy[n_train,1+48][8:10]+'.csv')
