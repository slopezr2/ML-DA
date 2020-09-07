import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

param = int(sys.argv[1])
print(param)
# Comment to run with GPU o Select CPU


pd.set_option('display.max_columns', None)

names = [ 'CnnMeteo_d',
    'CnvLstmMeteo',
    'CnvLstmMeteo_r_d'
]


mls_label = [ 'CnnMeteo_d',
	  'CnvLstmMeteo',
	  'CnvLstmMeteo_r_d'
]


#Parameters Tunning experiments 
semanas=[1,2]
LSTM_hidden_layers=[1,2]
cells=[100,200,500]

for semana in semanas:
	for n_LSTM_hidden_layers in LSTM_hidden_layers:
			for n_cells in cells:
				file_name="h_input_"+str(semana)+"_hidden_"+str(n_LSTM_hidden_layers)+"_cells_"+str(n_cells)+"_"+ mls_label[param]
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
				    plt.title(file_name)
				    plt.ylabel('root_mean_squared_error')
				    plt.xlabel('epoch')
				    plt.legend(['train', 'test'], loc='upper left')
				    plt.savefig(file_name)
				    plt.show()
				    

