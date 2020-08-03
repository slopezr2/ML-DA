import pandas as pd
import matplotlib.pyplot as plt
import json
pd.set_option('display.max_columns', None)

names = ['CnnMeteo', 'LstmVanillaMeteo', 'LstmStackedMeteo', 'LstmBidireccionalMeteo']
for i in range(0,4):
    with open('h_'+names[i]) as json_file:
        data = json.load(json_file)
        history = pd.DataFrame(data)
        history = history.reset_index()
        history['index2'] = history['index']
        history['index2'] = history.index2.astype(int)
        history = history.drop(['index'], axis=1)
        history.sort_values(by='index2', inplace=True)
        print(history.head(200))
        plt.plot(history['index2'], history['root_mean_squared_error'])
        plt.plot(history['index2'], history['val_root_mean_squared_error'])
        plt.title(names[i])
        plt.ylabel('root_mean_squared_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

