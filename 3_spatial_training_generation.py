import itertools
import math
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import distance

np.random.seed(333)
print 'working'
### In the folowing code, the station refers to the grids that PM2.5 is already labeled, doesnt constrant to monitoring stations
### and grid refers to the unlabeled one. 
def sim_feature(file, n_neighbor, n_feature):
	dat = pd.read_csv(file)
	### get the list of station/grid id
	id = dat['Station_ID'].unique()
	id.sort(axis=-1)
	n_station = len(id)
	### generate the dict, key is station (grid) id and value is its neighbors
	comb = itertools.combinations(id,len(id)-1)
	id = id[::-1]
	comb_target = itertools.combinations(id,1)
	dct = dict(zip(list(comb_target), list(comb)))
	### loop through all the possible combination and create the feature matrix
	dat_arr = dat.values
	data_sim = pd.DataFrame()
	### ID: a selected grid, nbr: its neighbor stations, land, road, dist are features
	ID = []
	nbr_pm = []
	land = []
	road = []
	inter = []
	dist = []
	row = 0
	for x in dct:
		comb_for_each = [m for m in itertools.combinations(dct[x],n_neighbor)]
		# comb_for_each = random.sample(comb_for_each, n_comb)
		s0 = dat_arr[dat_arr[:,0] == x]
		for xx in comb_for_each:
			row += 1
			ID.append(x[0])
			for xxx in xx:
				nbr_pm.append(xxx)
				s = dat_arr[dat_arr[:,0] == xxx]
				# f1 = distance.euclidean(s0[0][1:-4], s[0][1:-4])
				f1 = pearsonr(s0[0][1:-4], s[0][1:-4])[0]
				land.append(f1)
				f2 = float(s0[0][-4]/s[0][-4])
				road.append(f2)
				# f3 = float(s0[0][-3]/s[0][-3])
				# inter.append(f3)
				f4 = math.sqrt(math.pow((s0[0][-2] - s[0][-2]),2) + math.pow((s0[0][-1] - s[0][-1]),2))
				dist.append(f4)
	land = np.array(land).reshape([row,n_neighbor])
	road = np.array(road).reshape([row,n_neighbor])
	# inter = np.array(inter).reshape([row,n_neighbor])
	dist = np.array(dist).reshape([row,n_neighbor])
	nbr_pm = np.array(nbr_pm).reshape([row,n_neighbor])

	# f = np.hstack((nbr_pm, np.hstack((land, np.hstack((road, np.hstack((inter,dist))))))))
	f = np.hstack((nbr_pm, np.hstack((land, np.hstack((road,dist))))))

	### data_sim contains the similiarity between stations and grids
	data_sim['Station_ID'] = ID
	for i in range(n_feature*(n_neighbor)):
		data_sim[i] = f[:,i]
	return data_sim

def shuffle(file,size):
	dat = pd.read_csv(file)
	pm_arr = dat['PM2.5'].values
	pm_arr = pm_arr.reshape([13,dat.shape[0]/13]).transpose()
	np.random.seed(333)

	random_lst = np.random.randint(pm_arr.shape[0],size=size)
	pm_arr_random = pm_arr[random_lst,:]
	stations = dat['Station'].unique()
	pm_df = pd.DataFrame(pm_arr_random,columns= stations)
	return pm_df

def sim_pm(sim, pm, n_neighbor):
	sim_rep = sim.values.repeat(n_random,axis=0)
	pm_arr = pm.values
	station = sim.iloc[:,0].values

	PM_station = []
	nbr = sim.iloc[:,1:1+n_neighbor].values.astype(int)
	PM_nbr = np.zeros(shape=(sim_rep.shape[0],nbr.shape[1]))
	for i in range(nbr.shape[0]):
		for j in range(pm.shape[0]):
			s = station[i]
			PM_station.append(pm_arr[j,s])
			for k in range(nbr.shape[1]):
				n = nbr[i,k]
				PM_nbr[i*pm.shape[0]+j,k] = pm_arr[j,n]

	pm_sim = pd.DataFrame(sim_rep,columns=sim.columns)
	col_nbr = ["PM"+str(i+1) for i in range(n_neighbor)]
	
	for i in range(PM_nbr.shape[1]):
		pm_sim[col_nbr[i]] = PM_nbr[:,i]
	pm_sim['PM2.5'] = PM_station

	lst = ['Station_ID']
	for i in range(len(col_nbr)):
		lst.append(i)
	for x in col_nbr:
		lst.append(x)
	pm_sim = pm_sim.groupby(lst)

	pm_lst =[]
	for k, gp in pm_sim:
		mode = gp['PM2.5'].mode()
		try:
			pm_lst.append(mode[0])
		except IndexError:
			pm = gp.iloc[0,-1]
			pm_lst.append(pm)
	data = pm_sim.first()
	data['PM2.5'] = pm_lst
	data.to_csv('./Spatial_Data/test_corr.csv')
	return data

def forest(dat, n_neighbor):
	dat = dat.reset_index(drop=False)
	# print dat.columns
    #create the training & test sets, skipping the header row with [1:]
	np.random.seed(333)
	# dat = pd.read_csv('./Spatial_Data/test.csv')
	X = dat.iloc[:,n_neighbor+1:]

	data = np.random.permutation(X.values)
	data[:,n_neighbor+1:-1] = data[:,n_neighbor+1:-1] / data[:,n_neighbor+1:-1].max(axis=0)
	n = data.shape[0] * 0.7
	trainY = data[:n,-1]
	trainX = data[:n,:-1]
	testY = data[n:,-1]
	testX = data[n:,:-1]
    
	rf = RandomForestClassifier(n_estimators=100,max_depth=20,oob_score=True, warm_start=True)
	rf.fit(trainX, trainY)
	acc = rf.score(testX, testY)
	print acc
	print rf.feature_importances_
	# yest = rf.predict(testX)
	# pro = rf.predict_proba(testX)


if __name__ == '__main__':
	file_spatial = './Spatial_Data/spatial_feature_station.csv'
	file_pm = './Spatial_Data/PM2.5_Weather_3Cat_4.csv'
	n_feature = 4
	print 'number of features:', n_feature
	for	n_neighbor in [3]:
		np.random.seed(333)
		print 'number of neighbor:', n_neighbor
		for n_random in [1500]:
			print 'number of random:',n_random
			sim = sim_feature(file_spatial, n_neighbor, n_feature)
			pm = shuffle(file_pm,n_random)
			pm_sim = sim_pm(sim, pm, n_neighbor)
			forest(pm_sim, n_neighbor)
		