import pandas as pd

df = pd.read_csv('./SPEED/taxi_pm_std.csv')
df1 = pd.read_csv('./SPEED/taxi_pm_avg.csv')
df = df.fillna(method='pad')
df1 = df.fillna(method='pad')
df.to_csv('./SPEED/taxi_pm_std.csv')
df1.to_csv('./SPEED/taxi_pm_avg.csv')
