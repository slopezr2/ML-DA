import pandas as pd
df = pd.read_json (r'h_CnnMeteo.json')

df['loss'].plot()