# ML-DA
Repository for scripst of Machine Learning to Air Quality Forecasting and Data Asimilation

## Load Data
### Load All Data Files: 
```python
dataM = DataManager(path="data/") 
```

### Load some Data file:
```python
dataM = DataManager(path="data/", filter_items=["pm25"]) 
```

### Get Station 
```python
station3_pm25 = dataM.get_pm25("3")
```

## Machine Learning

### Pre-process data

```python
n_input_steps = 24*7*2
n_output_steps = 24*3
pre_processor = Combiner()
datax, datay = pre_processor.combine(n_input_steps, n_output_steps, station3_pm25.CONCENTRATION.values)
```


### Create Model
```python
n_train = 64*100
n_features = 1
X = datax[0:n_train, :]
Y = datay[0:n_train, :]
X = X.reshape((X.shape[0], X.shape[1],n_features))
cnnSiata = CnnSiata(n_input_steps,n_features, n_output_steps)
```
### Fit Model
```python
cnnSiata.model.fit(X, Y, epochs=5, verbose=1)
```
### Demonstrate prediction
```python
x_input = datax[n_train+10, :]
x_input = x_input.reshape((1, n_input_steps, n_features))
yhat = cnnSiata.model.predict(x_input, verbose=1)
```


## Plot Response
### Plot
```python
plt.plot(np.arange(0, n_input_steps), datax[n_train, :])
plt.plot(np.arange(n_input_steps,n_input_steps+n_output_steps), yhat[0,:],'r')
plt.plot(np.arange(n_input_steps,n_input_steps+n_output_steps), datay[n_train, :], 'g')
plt.show()
```
