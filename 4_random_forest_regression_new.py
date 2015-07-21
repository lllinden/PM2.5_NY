from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from scipy import stats
import sys
import timeit
print 'working'

def main():
	np.random.seed(333)
	w = str(5)
	n_br = str(4)
#	print 'total', 3,7
#	print 'neighbor',w,n_br
	train_0 = pd.read_csv('./Spatial_Data/train_trans_'+n_br+'_out_of_'+w+'_with_fuel.csv')
#	train_0 = pd.read_csv('./Spatial_Data/train_trans_8.csv')
	print train_0.shape[0]

	test_0 = pd.read_csv('./Spatial_Data/test_trans_'+n_br+'_out_of_'+w+'_with_fuel.csv')
#	test_0 = pd.read_csv('./Spatial_Data/test_trans_8.csv')
	print test_0.shape[0]

	train = train_0.iloc[:,5:].values
	test = test_0.iloc[:,5:].values

	train_random = np.random.permutation(train)
	train_random[:,4:-1] = train_random[:,4:-1] / train_random[:,4:-1].max(axis=0)
	test[:,4:-1] = test[:,4:-1] / test[:,4:-1].max(axis=0)

	trainY = train_random[:,-1]
	trainX = train_random[:,:-1]

	for n in [70]:
#	for d in np.arange(40,51,2):
#	for d in [32,34,36,44,46,48]:
		testY = test[:,-1]
		testX = test[:,:-1]

		start = timeit.default_timer()
		print 'tree number:', n
		print 'tree depth:', 24
		rf = RandomForestRegressor(n_estimators=n,max_depth=25,oob_score=True, random_state =333, n_jobs=10)
		rf.fit(trainX, trainY)
		r2 = rf.score(testX, testY)
		stop = timeit.default_timer()
		print 'r square:', r2
		print rf.feature_importances_

		time = test_0['Time']
		grid = test_0['Grid']
		result = pd.DataFrame()
		result['time']=time
		result['grid']=grid
		yest = rf.predict(testX)
		result['yest']=yest
		result['y'] = testY
		result = result.groupby([result['time'], result['grid']]).mean()
		yest = result['yest'].values
		ytest = result['y'].values
		ybar = np.mean(ytest)
			
		ssreg = np.sum((yest-ytest)**2)
		sstot = np.sum((ytest-ybar)**2)

		r2_trans =1 - ssreg/sstot
		print 'transformed r square:', r2_trans
		print 'time:', stop- start
		result.to_csv('test_result.csv')
#		result = np.vstack((yest, ytest))
#		np.savetxt('./Result/train_result_4_out_ot_5.csv', result.T, delimiter=',')	
		
if __name__=="__main__":
    main()
