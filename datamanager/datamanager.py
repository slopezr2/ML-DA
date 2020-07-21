import numpy as np
import pandas as pd
from datetime import datetime
from numpy import array
from sklearn.preprocessing import MinMaxScaler


class DataManager:
    def __init__(self, path, filter_items=[]):
        self.stations_pm25 = {}
        self.stations_no2 = {}
        self.stations_pm10 = {}
        self.stations_o3 = {}
        self.stations_so2 = {}
        self.stations_co = {}
        self.stations_Temperature = {}
        self.stations_Wind = {}
        self.SIATA_pm25 = pd.read_csv(path + 'SIATA_pm25.csv') if "pm25" in filter_items or len(filter_items) == 0 else None
        self.SIATA_no2 = pd.read_csv(path + 'SIATA_no2.csv') if "no2" in filter_items or len(filter_items) == 0 else None
        self.SIATA_pm10 = pd.read_csv(path + 'SIATA_pm10.csv') if "pm10" in filter_items or len(filter_items) == 0 else None
        self.SIATA_o3 = pd.read_csv(path + 'SIATA_o3.csv') if "o3" in filter_items or len(filter_items) == 0 else None
        self.SIATA_so2 = pd.read_csv(path + 'SIATA_so2.csv') if "so2" in filter_items or len(filter_items) == 0 else None
        self.SIATA_co = pd.read_csv(path + 'SIATA_co.csv') if "co" in filter_items or len(filter_items) == 0 else None
        self.SIATA_Temperature = pd.read_csv(path + 'Meteo_SIATA_Temperature_total_V2.csv', delimiter=';') if "temperature" in filter_items or len(filter_items) == 0 else None
        self.SIATA_Wind = pd.read_csv(path + 'Meteo_SIATA_Wind_total_V2.csv', delimiter=';') if "wind" in filter_items or len(filter_items) == 0 else None
        if (self.SIATA_Temperature is not None):
            print("Total Temperature "+str(len(self.SIATA_Temperature)))
            self.SIATA_Temperature['Date'] = pd.to_datetime(self.SIATA_Temperature[['YEAR', 'MONTH', 'DAY', 'HOUR']])

        if (self.SIATA_Wind is not None):
            print("Total Wind " + str(len(self.SIATA_Wind)))
            self.SIATA_Wind['Date'] = pd.to_datetime(self.SIATA_Wind[['YEAR', 'MONTH', 'DAY', 'HOUR']])

    def pre_process_station(self, siata, station, drop):
        station_temp = siata[siata['STATION'].values == station]
        station_temp['CONCENTRATION'].loc[station_temp['CONCENTRATION'] < 0] = np.NaN
        station_temp['CONCENTRATION'].loc[np.isnan(station_temp['CONCENTRATION'])] = np.nanmean(station_temp['CONCENTRATION'])
        if drop:
            station_temp.drop_duplicates(subset='Date', keep='last', inplace=True)
        station_temp.reset_index(inplace=True)
        return station_temp

    def pre_process_station_meteo(self, siata, station, drop):
        station_temp = siata[siata['station'].values == station]
        station_temp['Value'].loc[station_temp['Value'] < 0] = np.NaN
        station_temp['Value'].loc[np.isnan(station_temp['Value'])] = np.nanmean(station_temp['Value'])
        if drop:
            station_temp.drop_duplicates(subset='Date', keep='last', inplace=True)
        station_temp.reset_index(inplace=True)
        return station_temp

    def pre_process_all(self, station_number):
        self.stations_pm25[station_number] = self.pre_process_station(self.SIATA_pm25, "Station" + station_number,
                                                                      False)
        self.stations_pm10[station_number] = self.pre_process_station(self.SIATA_pm10, "Station" + station_number,
                                                                      False)
        self.stations_no2[station_number] = self.pre_process_station(self.SIATA_no2, "Station" + station_number, True)
        self.stations_o3[station_number] = self.pre_process_station(self.SIATA_o3, "Station" + station_number, True)
        self.stations_so2[station_number] = self.pre_process_station(self.SIATA_so2, "Station" + station_number, True)
        self.stations_co[station_number] = self.pre_process_station(self.SIATA_co, "Station" + station_number, True)
        self.stations_Temperature[station_number] = self.pre_process_station(self.SIATA_Temperature,
                                                                             "Station" + station_number, True)
        self.stations_Wind[station_number] = self.pre_process_station(self.SIATA_Wind, "Station" + station_number, True)

    def get_pm25(self, station_number):
        if not (station_number in self.stations_pm25):
            self.stations_pm25[station_number] = self.pre_process_station(self.SIATA_pm25, "Station" + station_number,
                                                                          True)
        return self.stations_pm25[station_number]

    def get_temperature(self, station_number):
        if not (station_number in self.stations_Temperature):
            self.stations_Temperature[station_number] = self.pre_process_station_meteo(self.SIATA_Temperature,
                                                                                 "station" + station_number, True)
        return self.stations_Temperature[station_number]

    def get_wind(self, station_number):
        if not (station_number in self.stations_Wind):
            self.stations_Wind[station_number] = self.pre_process_station_meteo(self.SIATA_Wind, "station" + station_number,
                                                                          True)
        return self.stations_Wind[station_number]


class Combiner:

    def create_dataset(self, dataset, look_back=1, step_forecast=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            if (i + look_back + step_forecast) > len(dataset):
                break
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back:i + look_back + step_forecast])
        return np.array(dataX), np.array(dataY)

    def split_sequences(self, sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        print(len(array(X)))
        print(len(array(y)))
        return array(X), array(y)

    def scale(self, data):
        scalerT = MinMaxScaler(feature_range=(0, 1))
        scalerT = scalerT.fit(data)
        return scalerT.transform(data), scalerT

    def combine(self, n_input_steps, n_output_steps, *argv):
        print(len(argv))
        self.scalers = []
        if len(argv) > 1:
            all_data = []
            for arg in argv:
                temp = arg.reshape((len(arg), 1))
                scaled, scaler = self.scale(temp)
                self.scalers.append(scaler)
                all_data.append(scaled)

            dataset = np.hstack(all_data)
            return self.split_sequences(dataset, n_input_steps, n_output_steps)
        else:
            return self.create_dataset(np.transpose( argv[0]), n_input_steps, n_output_steps)
