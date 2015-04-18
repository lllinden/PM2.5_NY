import numpy as np
import pandas as pd
from numpy import genfromtxt

data = pd.read_csv('pm_taxi_weather_turnstile.csv')

lst = []
for x in data['PM25C']:
	lst.append(x)

arr = np.array(lst)
arr[arr<5] = 1
arr[np.logical_and(arr>=5, arr<10)] = 2
arr[np.logical_and(arr>=10, arr<15)] = 3
arr[np.logical_and(arr>=15, arr<20)] =4
arr[np.logical_and(arr>=20, arr<25)] = 5
arr[arr>=25] = 6

unique, counts = np.unique(arr, return_counts=True)
print unique, counts

data['PM_Cate'] = arr
# data.to_csv('pm_taxi_weather_turnstile_1.c/sv')
