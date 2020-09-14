import numpy as np
import pandas as pd
from datetime import datetime
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


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
        self.SIATA_pm25 = pd.read_csv(path + 'SIATA_pm25.csv') if "pm25" in filter_items or len(
            filter_items) == 0 else None
        self.SIATA_no2 = pd.read_csv(path + 'SIATA_no2.csv') if "no2" in filter_items or len(
            filter_items) == 0 else None
        self.SIATA_pm10 = pd.read_csv(path + 'SIATA_pm10.csv') if "pm10" in filter_items or len(
            filter_items) == 0 else None
        self.SIATA_o3 = pd.read_csv(path + 'SIATA_o3.csv') if "o3" in filter_items or len(filter_items) == 0 else None
        self.SIATA_so2 = pd.read_csv(path + 'SIATA_so2.csv') if "so2" in filter_items or len(
            filter_items) == 0 else None
        self.SIATA_co = pd.read_csv(path + 'SIATA_co.csv') if "co" in filter_items or len(filter_items) == 0 else None
        self.SIATA_Temperature = pd.read_csv(path + 'Meteo_SIATA_Temperature_total_V2.csv',
                                             delimiter=';') if "temperature" in filter_items or len(
            filter_items) == 0 else None
        self.SIATA_Wind = pd.read_csv(path + 'Meteo_SIATA_Wind_total_V2.csv',
                                      delimiter=';') if "wind" in filter_items or len(filter_items) == 0 else None
        if (self.SIATA_Temperature is not None):
            print("Total Temperature " + str(len(self.SIATA_Temperature)))
            self.SIATA_Temperature['Date'] = pd.to_datetime(self.SIATA_Temperature[['YEAR', 'MONTH', 'DAY', 'HOUR']])

        if (self.SIATA_Wind is not None):
            print("Total Wind " + str(len(self.SIATA_Wind)))
            self.SIATA_Wind['Date'] = pd.to_datetime(self.SIATA_Wind[['YEAR', 'MONTH', 'DAY', 'HOUR']])

    def pre_process_station(self, siata, station, drop):
        station_temp = siata[siata['STATION'].values == station]
        station_temp['CONCENTRATION'].loc[station_temp['CONCENTRATION'] < 0] = np.NaN
        station_temp['CONCENTRATION'].loc[np.isnan(station_temp['CONCENTRATION'])] = np.nanmean(
            station_temp['CONCENTRATION'])
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
            self.stations_Wind[station_number] = self.pre_process_station_meteo(self.SIATA_Wind,
                                                                                "station" + station_number,
                                                                                True)
        return self.stations_Wind[station_number]

    def generate_ensambles(self, xb, std, n_ensembles):
        temp = np.zeros((xb.size, n_ensembles))
        for i in range(n_ensembles):
            temp[:, i] = np.add(xb.T, std * np.random.randn(xb.size, ))
        return temp

    def calculate_cov(self, X):
        return np.cov(X)

    def calculate_cov2(self, X):
        xb = np.zeros((X[:, 0].size, 1))
        xb[:, 0] = np.mean(X, axis=1)
        temp = X - xb.dot(np.ones((X[0, :].size, 1)).T)
        return 1 / X[:, 0].size * temp.dot(temp.T)


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
            return self.create_dataset(np.transpose(argv[0]), n_input_steps, n_output_steps)


class CsvWriter:

    def __init__(self, parameter, unit, avg):
        self.data = {'STATION': [], 'LAT': [], 'LON': [], 'CONCENTRATION': [], 'YEAR': [], 'MONTH': [], 'DAY': [],
                     'HOUR': []}
        self.parameter = parameter
        self.unit = unit
        self.avg = avg
        template = 'observations/templates/Observaciones_SIATA_tpm25_20190101.csv'
        self.template = pd.read_csv(template)

    def add(self, yhat, station, dateini, hours):
        self.data['CONCENTRATION'].extend(yhat)
        self.data['STATION'].extend([station for i in yhat])
        dates = self.daterange(dateini, hours, len(yhat))
        self.data['YEAR'].extend([d.year for d in dates])
        self.data['MONTH'].extend([d.month for d in dates])
        self.data['DAY'].extend([d.day for d in dates])
        self.data['HOUR'].extend([d.hour for d in dates])
        lon = self.template[self.template['STATION'] == station]['LON'].values[0]
        lat = self.template[self.template['STATION'] == station]['LAT'].values[0]
        self.data['LON'].extend([lon for i in yhat])
        self.data['LAT'].extend([lat for i in yhat])

    def daterange(self, start_date, hour, n):
        delta = timedelta(hours=hour)
        for i in range(n):
            yield start_date
            start_date += delta

    def write_observation(self, path):
        self.data['PARAMETER'] = [self.parameter for i in range(len(self.data['STATION']))]
        self.data['UNITS'] = [self.unit for i in range(len(self.data['STATION']))]
        self.data['AVERAGING_PERIOD'] = [self.avg for i in range(len(self.data['STATION']))]
        pd.DataFrame.from_dict(self.data).to_csv(path, index=False)
