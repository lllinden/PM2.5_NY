import pandas as pd
import numpy as np
import timeit, random	
import itertools
import sys
import json

np.random.seed(333)
random.seed(333)

def sim_3d_mtr():
	start = timeit.default_timer()
	landfile = './Spatial_Data/land.csv'
	roadfile = './Spatial_Data/road.csv'
	distfile = './Spatial_Data/dist.csv'
	fuelfile = './Spatial_Data/fuel.csv'
	dat_land = np.genfromtxt(landfile, delimiter=',')
	dat_road = np.genfromtxt(roadfile, delimiter=',')
	dat_dist = np.genfromtxt(distfile, delimiter=',')
	dat_fuel = np.genfromtxt(fuelfile, delimiter=',')
	sim_mtr = np.dstack((np.dstack((np.dstack((dat_land, dat_road)), dat_dist)),dat_fuel))
	stop = timeit.default_timer()
	print 'sim_3_mtr, time:', stop-start  
	return sim_mtr
def trans(pm_df, station_dict):
	start = timeit.default_timer()
	pm_arr = pm_df.values
	ID = []
	nbr = []
	pm = []
	nbr_pm = []
	land =[]
	road = []
	dist = []
	time = []
	fuel =[]
	row = 0
	pm_sim = pd.DataFrame()

	# for t in range(pm_arr.shape[0]):
	# 	pm_t = pm_arr[t,:]
	# 	for s_y in station_dict:
	# 		ID.append(s_y)
	# 		g_id.append(s_y)
	# 		pm.append(pm_t[int(s_y)])
	# 		time.append(t)
	# 		row += 1
	# 		for s_x in station_dict[s_y]:
	# 			nbr.append(s_x)
	# 			nbr_pm.append(pm_t[int(s_x)])
	# 			land.append(sim_mtr[s_y,s_x,0])
	# 			road.append(sim_mtr[s_y,s_x,1])
	# 			dist.append(sim_mtr[s_y,s_x,2])

	for t in range(pm_arr.shape[0]):
		pm_t = pm_arr[t,:]
		for s_y in station_dict:
			s_X = station_dict[s_y][:n]
			nbr_comb = [x for x in itertools.combinations(s_X, n_nbr)]
			for ss_x in nbr_comb:
				ID.append(s_y)
				time.append(t)
				row += 1
#				pm.append(pm_t[int(s_y)])
				for s_x in ss_x:
					nbr.append(s_x)
					nbr_pm.append(pm_t[int(s_x)])
					land.append(sim_mtr[s_y,s_x,0])
					road.append(sim_mtr[s_y,s_x,1])
					dist.append(sim_mtr[s_y,s_x,2])
					fuel.append(sim_mtr[s_y,s_x,3])

	nbr = np.array(nbr).reshape([row,n_nbr])
	land = np.array(land).reshape([row,n_nbr])
	road = np.array(road).reshape([row,n_nbr])
	dist = np.array(dist).reshape([row,n_nbr])
	fuel = np.array(fuel).reshape([row,n_nbr])
	
	nbr_pm = np.array(nbr_pm).reshape([row,n_nbr])

	dat = np.hstack((nbr, np.hstack((nbr_pm, np.hstack((land, np.hstack((road,np.hstack((dist,fuel))))))))))
	pm_sim['Time'] = time
	pm_sim['Station'] = ID
	for i in range((n_feat+1)*(n_nbr)):
		pm_sim[i] = dat[:,i]
#	pm_sim['PM2.5'] = pm

	c_n = ['n'+str(i+1) for i in range(n_nbr)]
	c_p = ["pm"+str(i+1) for i in range(n_nbr)]
	c_l = ['l'+str(i+1) for i in range(n_nbr)]
	c_r = ['r'+str(i+1) for i in range(n_nbr)]
	c_d = ['d'+str(i+1) for i in range(n_nbr)]
	c_f = ['f'+str(i+1) for i in range(n_nbr)]
	pm_sim.columns = ['Time','Grid']+ c_n + c_p + c_l + c_r + c_d + c_f# + ['PM2.5']

	stop = timeit.default_timer()
	print 'feature_trans, time:', stop-start

	return pm_sim	

if __name__ == '__main__':
	file = './All_station_2014_nu.csv'
	file1 = './data_nu.csv'
	pm_df = pd.read_csv(file)
	pm_df_unlabel = pd.read_csv(file1)

	with open("result_station_train.json") as json_file:
		train_station = json.load(json_file)
	with open('result_station_test.json') as json_file:
		test_station = json.load(json_file)
	with open('dist_grids.json') as json_file:
		unlabled_grid = json.load(json_file)	

	n_nbr = 4
	n = 5
	n_feat = 5
	sim_mtr = sim_3d_mtr()

	# train = trans(pm_df, train_station)
	# train.to_csv('./Spatial_Data/train_trans_'+str(n_nbr)+'_out_of_'+str(n)+'_with_fuel.csv', index=False)
	# test = trans(pm_df, test_station)
	# test.to_csv('./Spatial_Data/test_trans_'+str(n_nbr)+'_out_of_'+str(n)+'_with_fuel.csv',index=False)
	unlabel = trans(pm_df_unlabel, unlabled_grid)
	unlabel.to_csv('./Spatial_Data/unlabel_trans'+str(n_nbr)+'_out_of_'+str(n)+'_with_fuel.csv',index=False)

