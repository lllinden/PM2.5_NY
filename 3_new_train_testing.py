import pandas as pd
import numpy as np
import timeit, random	
import itertools
import sys

np.random.seed(333)


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

def feature_trans(pm_df,dir):
	start = timeit.default_timer()

	pm_arr = pm_df.values


	id = map(int, list(pm_df.columns))
	n_station = len(train_station)

	comb = itertools.combinations(train_station,n_station-1)
	comb_target = itertools.combinations(train_station[::-1],1)
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

def test_trans(pm_df,dir):
	start = timeit.default_timer()
	pm_df = pm_df.values
	g_id = []
	nbr_id = []
	nbr_pm = []
	time = []
	land = []
	road = []
	dist = []
	row = 0
	pm=[]
	test_trans = pd.DataFrame()
	for i in range(pm_df.shape[0]):
		for x in test_station:
			pair_station = np.delete(station,x)
			nbr_comb = [m for m in itertools.combinations(pair_station, n_nbr)]
			nbr_comb = random.sample(nbr_comb, 10)
			for nbrs in nbr_comb:
				g_id.append(x)
				row+=1
				time.append(i)
				pm.append(pm_df[i,x])
				for nbr in nbrs:
					nbr_id.append(nbr)
					nbr_pm.append(pm_df[i,nbr])
					land.append(sim_mtr[x, nbr, 0])
					road.append(sim_mtr[x, nbr, 1])
					dist.append(sim_mtr[x, nbr, 2])
	nbr_id = np.array(nbr_id).reshape([row,n_nbr])
	nbr_pm = np.array(nbr_pm).reshape([row,n_nbr])
	road = np.array(road).reshape([row,n_nbr])
	dist = np.array(dist).reshape([row,n_nbr])
	land = np.array(land).reshape([row,n_nbr])

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
	test_trans['PM2.5'] = pm
	test_trans.to_csv('./Spatial_Data/Self_Training/'+dir+'.csv', index=False)

	stop = timeit.default_timer()
	print 'test_trans, time:', stop-start

	return test_trans	

def unlabled_trans(pm_df, dir):
	start = timeit.default_timer()
	grid_id = list(range(13,1024))
	station_id = train_station
	pm_df = np.random.permutation(pm_df.values)
	length = pm_df.shape[0]
	pm_df = pm_df[:length/2,:]

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

	for i in range(pm_df.shape[0]):
		for grid in grid_id:
			nbr_comb = random.sample(nbr_comb, 10)
			for nbrs in nbr_comb:
				g_id.append(grid)
				row+=1
				time.append(i)
				for nbr in nbrs:
					nbr_id.append(nbr)
					nbr_pm.append(pm_df[i,nbr])
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
	test_trans.to_csv('./Spatial_Data/Self_Training/unlabel'+sys.argv[1]+'_1.csv', index=False)

	stop = timeit.default_timer()
	print 'test_trans, time:', stop-start

	return test_trans

if __name__ == '__main__':
	n_month = sys.argv[1]
	file = './Spatial_Data/PM2.5_'+n_month+'_month.csv'
	dat_station_pm = pd.read_csv(file)
	station = np.arange(13)
	train_station = np.random.choice(13, 9,replace=False)
	test_station = np.delete(station, train_station)

	n_nbr = 3
	n_feat = 4

	sim_mtr = sim_3d_mtr()
	feature_trans(dat_station_pm, 'train_'+n_month)
	test_trans(dat_station_pm, 'test_'+n_month)
	unlabled_trans(dat_station_pm, 'test_'+n_month)

