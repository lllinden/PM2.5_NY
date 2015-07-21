import pandas as pd
import numpy as np
import json

file = './Spatial_Data/dist_kilo.csv'
dat = pd.read_csv(file,header=None).iloc[13:,:]

 
keys = list(dat.index.tolist())
dct = {key: [] for key in keys}

for i in range(dat.shape[0]):
	sort_dist = np.argsort(dat.iloc[i,:]).tolist()
	dct[dat.index[i]] = sort_dist
	# print sort_dist


with open('dist_grids.json', 'w') as fp:
    json.dump(dct, fp)