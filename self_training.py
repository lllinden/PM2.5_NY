import timeit
import itertools
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(333)
random.seed(333)
def sim_3d_mtr():
	# load the road, landuse and distance matrix of grids, and dstack into a 3-D array
	start = timeit.default_timer()

	landfile = './Spatial_Data/land.csv'
	roadfile = './Spatial_Data/road.csv'
	distfile = './Spatial_Data/dist.csv'
	dat_land = np.genfromtxt(landfile, delimiter=',')
	dat_road = np.genfromtxt(roadfile, delimiter=',')
	dat_dist = np.genfromtxt(distfile, delimiter=',')
	sim_mtr = np.dstack((np.dstack((dat_land, dat_road)), dat_dist))
	stop = timeit.default_timer()
	print 'sim_3_mtr, time:', stop-start  
	
	return sim_mtr

def train_test():
	start = timeit.default_timer()
	
	file = './Spatial_Data/PM2.5_All.csv'
	# rng = pd.date_range('6/1/2014', periods=7296, freq='H')
	# pm = pd.read_csv(file).set_index()
	pm = pd.read_csv(file)
	time_len = pm.shape[0]
	rows = np.random.choice(pm.index.values, time_len)
	unlabeled = pm.ix[rows[:time_len*0.2]]
	valid = pm.ix[rows[time_len*0.2:time_len*0.4]]
	labeled = pm.ix[rows[time_len*0.4:]]
	
	stop = timeit.default_timer()
	print 'train_test, time:', stop-start  

	return labeled, unlabeled, valid
	# load the data for training and testing

def feature_trans(sim_mtr, pm_df,dir):
	start = timeit.default_timer()

	pm_arr = pm_df.values
	id = map(int, list(pm_df.columns))
	n_station = len(id)
	comb = itertools.combinations(id,len(id)-1)
	comb_target = itertools.combinations(id[::-1],1)
	dct = dict(zip(list(comb_target), list(comb)))
	### ID: station id, nbr: neighbor id, pm: station pm, nbr_pm: neighbor pm, land,road,dist: the similiarity between the station and neighbors. 
	ID = []
	nbr = []
	pm = []
	nbr_pm = []
	land =[]
	road = []
	dist = []
	row = 0
	pm_sim = pd.DataFrame()

	for i in range(pm_arr.shape[0]):
		pm_vec = pm_arr[i,:]
		for x in dct:
			comb_for_each = [m for m in itertools.combinations(dct[x],n_nbr)]
			x = x[0]
			for xx in comb_for_each:
				row += 1
				ID.append(x)
				pm.append(pm_vec[x])
				for xxx in xx:
					nbr.append(xxx)
					nbr_pm.append(pm_vec[xxx])
					land.append(sim_mtr[x,xxx,0])
					road.append(sim_mtr[x,xxx,1])
					dist.append(sim_mtr[x,xxx,2])

	nbr = np.array(nbr).reshape([row,n_nbr])
	land = np.array(land).reshape([row,n_nbr])
	road = np.array(road).reshape([row,n_nbr])
	dist = np.array(dist).reshape([row,n_nbr])
	nbr_pm = np.array(nbr_pm).reshape([row,n_nbr])

	dat = np.hstack((nbr, np.hstack((nbr_pm, np.hstack((land, np.hstack((road,dist))))))))
	pm_sim['Station'] = ID
	for i in range((n_feat+1)*(n_nbr)):
		pm_sim[i] = dat[:,i]
	pm_sim['PM2.5'] = pm

	c_n = ['n'+str(i+1) for i in range(n_nbr)]
	c_p = ["pm"+str(i+1) for i in range(n_nbr)]
	c_l = ['l'+str(i+1) for i in range(n_nbr)]
	c_r = ['r'+str(i+1) for i in range(n_nbr)]
	c_d = ['d'+str(i+1) for i in range(n_nbr)]
	pm_sim.columns = ['Station']+ c_n + c_p + c_l + c_r + c_d + ['PM2.5']

	lst = ['Station']
	for x in c_n:
		lst.append(x)
	for x in c_p:
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
	data.to_csv('./Spatial_Data/Self_Training/'+dir+'.csv')


	stop = timeit.default_timer()
	print 'feature_trans, time:', stop-start

def test_trans(sim_mtr, test):
	start = timeit.default_timer()
	grid_id = list(range(13,1024))
	station_id = list(range(13))
	test_arr = test.values
	# nbr_comb = itertools.combinations(station_id,3)
	nbr_comb = [x for x in itertools.combinations(station_id, 3)]

	### ID: station id, nbr: neighbor id, pm: station pm, nbr_pm: neighbor pm, land,road,dist: the similiarity between the station and neighbors. 
	g_id = []
	nbr_id = []
	nbr_pm = []
	time = []
	land = []
	road = []
	dist = []
	row = 0
	test_trans = pd.DataFrame()

	for i in range(test.shape[0]):
		for grid in grid_id:
			nbr_comb = random.sample(nbr_comb, 10)
			for nbrs in nbr_comb:
				g_id.append(grid)
				row+=1
				time.append(test.index[i])
				for nbr in nbrs:
					nbr_id.append(nbr)
					nbr_pm.append(test_arr[i,nbr])
					land.append(sim_mtr[grid, nbr, 0])
					road.append(sim_mtr[grid, nbr, 1])
					dist.append(sim_mtr[grid, nbr, 2])

	nbr_id = np.array(nbr_id).reshape([row,n_nbr])
	nbr_pm = np.array(nbr_pm).reshape([row,n_nbr])
	land = np.array(land).reshape([row,n_nbr])
	road = np.array(road).reshape([row,n_nbr])
	dist = np.array(dist).reshape([row,n_nbr])

	dat = np.hstack((nbr_id, np.hstack((nbr_pm, np.hstack((land, np.hstack((road,dist))))))))

	test_trans['Time'] = time
	test_trans['Grid'] = g_id

	for i in range((n_feat+1)*(n_nbr)):
		test_trans[i] = dat[:,i]

	c_n = ['n'+str(i+1) for i in range(n_nbr)]
	c_p = ["pm"+str(i+1) for i in range(n_nbr)]
	c_l = ['l'+str(i+1) for i in range(n_nbr)]
	c_r = ['r'+str(i+1) for i in range(n_nbr)]
	c_d = ['d'+str(i+1) for i in range(n_nbr)]
	test_trans.columns = ['Time','Grid']+ c_n + c_p + c_l + c_r + c_d 
	test_trans.to_csv('./Spatial_Data/Self_Training/unlabel.csv')


	stop = timeit.default_timer()
	print 'test_trans, time:', stop-start

	return test_trans

def self_training_data(sim_mtr, test, valid, result, pm_df, iter):
	start = timeit.default_timer()
	train = pd.read_csv('./Spatial_Data/Self_Training/train.csv')
	print 'old_train:', train.shape

	result = result.groupby([result[0], result[1]]).mean()
	result['max'] = result.max(axis=1)
	result['PM'] = result.idxmax(axis=1)
	result = result.drop([2,3,4], axis=1).reset_index().sort('max', ascending=False)

	label = result.iloc[:50,:]
	station_id = []
	nbr_id = []
	nbr_pm = []
	land = []
	road = []
	dist = []
	pm = []
	row = 0
	new_train = pd.DataFrame()

	nbr_comb = [x for x in itertools.combinations(list(range(13)), 3)]

	for i in range(label.shape[0]):
		for nbrs in nbr_comb:
			station_id.append(int(label.iloc[i,1]))
			pm.append(int(label.iloc[i,-1]-2))
			row += 1
			for nbr in nbrs:
				nbr_id.append(nbr)
				land.append(sim_mtr[int(label.iloc[i,1]), nbr, 0])
				road.append(sim_mtr[int(label.iloc[i,1]), nbr, 1])
				dist.append(sim_mtr[int(label.iloc[i,1]), nbr, 2])
				nbr_pm.append(pm_df.ix[int(label.iloc[i,0])][nbr])
	n_nbr = 3
	nbr_id = np.array(nbr_id).reshape([row,n_nbr])
	nbr_pm = np.array(nbr_pm).reshape([row,n_nbr])
	land = np.array(land).reshape([row,n_nbr])
	road = np.array(road).reshape([row,n_nbr])
	dist = np.array(dist).reshape([row,n_nbr])

	dat = np.hstack((nbr_id, np.hstack((nbr_pm, np.hstack((land, np.hstack((road,dist))))))))

	new_train['Station'] = station_id
	for i in range(15):
		new_train[i] = dat[:,i]
	new_train['PM2.5'] = pm

	c_n = ['n'+str(i+1) for i in range(n_nbr)]
	c_p = ["pm"+str(i+1) for i in range(n_nbr)]
	c_l = ['l'+str(i+1) for i in range(n_nbr)]
	c_r = ['r'+str(i+1) for i in range(n_nbr)]
	c_d = ['d'+str(i+1) for i in range(n_nbr)]
	new_train.columns = ['Station']+ c_n + c_p + c_l + c_r + c_d + ['PM2.5']

	train_added = train.append(new_train)
	train_added.to_csv('./Spatial_Data/Self_Training/Train/train.csv', index=False)

	lst = ['Station']
	for x in c_n:
		lst.append(x)
	for x in c_p:
		lst.append(x)
	train_added = train_added.groupby(lst)

	pm_lst =[]
	for k, gp in train_added:
		if len(gp) > 1: 
			mode = gp['PM2.5'].mode()
			try:
				pm_lst.append(mode[0])
			except IndexError:
				pm = gp.iloc[0,-1]
				pm_lst.append(mode)
		else:
			pm_lst.append(gp['PM2.5'])

	print len(pm_lst)
	train_added = train_added.first().reset_index()
	train_added['PM2.5'] = pm_lst

	print 'new_train:', train_added.shape


	stop = timeit.default_timer()
	print 'time:', stop-start
	# rf(train_new, test_new, valid, iter)

def rf(dat, dat_test, valid, iter):
	iter += 1
	start = timeit.default_timer()
	dat_test = dat_test.drop('Unnamed: 0',1)

	data = np.random.permutation(dat.iloc[:,4:].values)
	data1 = np.random.permutation(valid.iloc[:,4:].values)
	data2 = np.random.permutation(dat_test.iloc[:,5:].values)

	data[:,4:-1] = data[:,4:-1] / data[:,4:-1].max(axis=0)
	data1[:,4:-1] = data1[:,4:-1] / data1[:,4:-1].max(axis=0)
	data2[:,4:-1] = data2[:,4:-1] / data2[:,4:-1].max(axis=0)

	trainX = data[:,:-1]
	trainy = data[:,-1]
	validX = data1[:,:-1]
	validy = data1[:,-1]
	testX = data2


	rf = RandomForestClassifier(n_estimators=100,max_depth=20,oob_score=True, warm_start=True)
	rf.fit(trainX, trainy)
	acc = rf.score(validX, validy)
	print 'random forest accuracy:', acc
	print rf.feature_importances_
	
	time = dat_test.iloc[:,0].values
	grid = dat_test.iloc[:,1].values
	
	pro = rf.predict_proba(testX)
	r = np.vstack((time,grid)).T
	result = np.hstack((r, pro))
	np.savetxt("./Spatial_Data/Self_Training/result.csv", result, delimiter=",")
	stop = timeit.default_timer()
	print 'rf time:', stop - start

	# if iter >= 2:
	# 	return result
	# else:
	# 	self_training_data(dat, dat_test, valid, result, iter)

if __name__ == '__main__':
	n_nbr = 3
	n_feat = 4
	iter = 0
	sim_mtr = sim_3d_mtr()
	
	### splite the training, testing and validation dataset.
	train_pm, test_pm, valid_pm = train_test()
	
	### for each time point in training data, generate the combination between stations and pair it with the similiarities. 
	# train_dat = feature_trans(sim_mtr,train_pm,'train')
	# test_dat = feature_trans(sim_mtr, test_pm,'test')
	# valid_dat = feature_trans(sim_mtr, valid_pm,'valid')
	## for each time point, generating 20 pairs of similiarity for each grid
	# test_trans = test_trans(sim_mtr, test_pm)


	# training = pd.read_csv('./Spatial_Data/Self_Training/train.csv')
	testing = pd.read_csv('./Spatial_Data/Self_Training/unlabel.csv')
	valid = pd.read_csv('./Spatial_Data/Self_Training/valid.csv')
	# output = rf(training, testing, valid, iter)
	result = pd.read_csv('./Spatial_Data/Self_Training/result.csv', header=None)
	self_training_data(sim_mtr, testing, valid, result, test_pm, iter)
