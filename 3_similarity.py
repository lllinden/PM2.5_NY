import itertools
import math
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
c = 3
lst = [1,2,6,8,9,10,11,12,14]
comb = itertools.combinations(lst,len(lst)-1)
lst.reverse()
comb_target = itertools.combinations(lst,1)
dct = dict(zip(list(comb_target), list(comb)))

file = './Spatial_Data/sim_data.csv'
dat = pd.read_csv(file)
dat_arr = dat.values
data = pd.DataFrame()
ID = []
land = []
road = []
dist = []
air = []
PM = []

for x in dct:
	comb_for_each = itertools.combinations(dct[x],c)
	s0 = dat_arr[dat_arr[:,1] == x]
	for xx in comb_for_each:
		PM.append(s0[0][-1])
		ID.append(x[0])
		# print j
		for xxx in xx:
			s = dat_arr[dat_arr[:,1] == xxx]
			air.append(s[0][-1])
			f1 = pearsonr(s0[0][2:-4], s[0][2:-4])[0]
			land.append(f1)
			f2 = float(s0[0][-4]/s[0][-4])
			road.append(f2)
			f3 = math.sqrt(math.pow((s0[0][-3] - s[0][-3]),2) + math.pow((s0[0][-2] - s[0][-2]),2))
			dist.append(f3)

print len(land),len(road), len(dist)
r = len(PM)
land = np.array(land).reshape([r,c])
road = np.array(road).reshape([r,c])
dist = np.array(dist).reshape([r,c])
air = np.array(air).reshape([r,c])
f = np.hstack((air,np.hstack((land, np.hstack((road,dist))))))

data['Station_ID'] = ID
data['PM2.5'] = PM
# col = ['pm1','pm2','pm3','l1','l2','l3','r1','r2','r3','d1','d2','d3']
for i in range(4*(c)):
	data[i] = f[:,i]

data.to_csv('./Spatial_Data/sim_4_1.csv')