import numpy as np
import pandas as pd

np.random.seed(333)

file = './Spatial_Data/PM2.5_All.csv'

# rng = pd.date_range('6/1/2014', periods=7296, freq='H')
# pm = pd.read_csv(file).set_index()
pm = pd.read_csv(file)

time_len = pm.shape[0]
rows = np.random.choice(pm.index.values, time_len)
labeled = pm.ix[rows[time_len*0.2:]]
unlabeled = pm.ix[rows[:time_len*0.2]]
