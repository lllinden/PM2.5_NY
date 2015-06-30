import pandas as pd
import numpy as np
from scipy.spatial import distance

file = './Spatial_Data/Grid_Land_Use.csv'
dat = pd.read_csv(file)
# print dat.head(2)

land = dat.values[:,1:12]
land_corr = np.corrcoef(land)

road = dat.values[:,13]
road_ratio = np.zeros([road.shape[0], road.shape[0]])
for i in range(road.shape[0]):
	for j in range(road.shape[0]):
		road_ratio[i,j] = road[i]/road[j]

dist =  dat.values[:,-2:]
dist_eucl = np.zeros([road.shape[0], road.shape[0]])
for i in range(road.shape[0]):
	for j in range(road.shape[0]):
				dist_eucl[i,j] = distance.euclidean(dist[i,:], dist[j,:])

np.savetxt("land.csv", land_corr, delimiter=",")
np.savetxt("road.csv", road_ratio, delimiter=",")
np.savetxt('dist.csv', dist_eucl, delimiter=",")

