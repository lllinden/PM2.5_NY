from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from scipy import stats
import sys
import timeit
print 'working'

def main():
	np.random.seed(33)
	train_0 = pd.read_csv('./Spatial_Data/train_trans_4_out_of_5_with_fuel.csv')
#	print train_0.shape[0]
	test_0 = pd.read_csv('./Spatial_Data/unlabel_trans_4_out_of_5_with_fuel.csv')
#	print test_0.shape[0]
	train = train_0.iloc[:,5:].values
	test= test_0.iloc[:,5:].values

	train_random = np.random.permutation(train)
	train_random[:,4:-1] = train_random[:,4:-1] / train_random[:,4:-1].max(axis=0)
	test[:,4:] = test[:,4:] / test[:,4:].max(axis=0)

	trainY = train_random[:,-1]
	trainX = train_random[:,:-1]
	
	start = timeit.default_timer()
	rf = RandomForestRegressor(n_estimators=100, max_depth=50, oob_score=True, random_state =333, n_jobs=15)
	rf.fit(trainX, trainY)

	stop = timeit.default_timer()

	print rf.feature_importances_

	yest = rf.predict(test)		
	time = test_0.iloc[:,0].values
 	grid = test_0.iloc[:,1].values
	result = pd.DataFrame()
	result['Time'] = time
	result['Grid'] = grid
	result['yest'] = yest
	result = result.groupby([result['Time'], result['Grid']]).mean()
	result.to_csv('./predict/unlabeled_4_5_with_fuel.csv')	


if __name__=="__main__":
    main()
