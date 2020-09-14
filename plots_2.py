import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import numpy as np


# Comment to run with GPU o Select CPU


pd.set_option('display.max_columns', None)


#
stations=pd.read_csv('./data/Lista_estaciones.csv',delimiter=',')
stations_SIATA=np.array(stations['PM25'].values).astype('str')
stations_Meteo=np.array(stations['Meteo'].values).astype('str')
for station in range(len(stations_SIATA)):
    file_name="./history_files/h_Station"+stations_SIATA[station] 
    with open(file_name) as json_file:
        data = json.load(json_file)
    history = pd.DataFrame(data)
    history = history.reset_index()
    history['index2'] = history['index']
    history['index2'] = history.index2.astype(int)
    history = history.drop(['index'], axis=1)
    history.sort_values(by='index2', inplace=True)
    print(history.head(200))
    history = history.head(300)
    plt.plot(history['index2'], history['root_mean_squared_error'])
    plt.plot(history['index2'], history['val_root_mean_squared_error'])
    plt.title("Station"+stations_SIATA[station])
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./figures/h_Station"+stations_SIATA[station])
    plt.show()
				    

