from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from scipy import stats
import sys

print 'working'

def main():
	np.random.seed(33)
	train_0 = pd.read_csv('./Spatial_Data/train_summer_trans.csv')
	print train_0.shape[0]
	test_0 = pd.read_csv('./Spatial_Data/test_summer_trans.csv')
	print test_0.shape[0]
	train = train_0.iloc[:,4:].values
	test = test_0.iloc[:,5:].values

	train_random = np.random.permutation(train)
	train_random[:,4:-1] = train_random[:,4:-1] / train_random[:,4:-1].max(axis=0)
	test[:,4:-1] = test[:,4:-1] / test[:,4:-1].max(axis=0)



	trainY = train_random[:,-1]
	trainX = train_random[:,:-1]
	testY = test[:,-1]
	testX = test[:,:-1]
	rf = RandomForestRegressor(n_estimators=40,max_depth=10,oob_score=True)
	rf.fit(trainX, trainY)
	r2 = rf.score(testX, testY)
	print r2
	# print rf.feature_importances_
	# print rf.estimators_
	yest = rf.predict(testX)

	yest = yest.reshape(([yest.shape[0]/84,84]))
	yest = stats.mean(yest.T)

	testY = testY.reshape(([testY.shape[0]/84,84]))
	testY = stats.mean(testY.T)

	ybar = np.sum(testY)/len(testY)
	ssreg = np.sum((yest-ybar)**2)
	sstot = np.sum((testY-ybar)**2)

	r2_trans = ssreg/sstot
	print r2_trans

	# yest = rf.predict(predict)
	# yest = yest.reshape(([yest.shape[0]/10,10]))
	# yest = stats.mode(yest.T)[0].T


# 	predict_0 = pd.read_csv('./Prediction/unlabel_4c_trans_84s.csv')
# 	predict = predict_0.iloc[:,5:].values
# 	predict[:,4:-1] = predict[:,4:-1] / predict[:,4:-1].max(axis=0)

# 	time = predict_0.iloc[:,0].values
# 	grid = predict_0.iloc[:,1].values
# 	##yest = rf.predict(predict)
# 	pro_test = rf.predict_proba(predict)
# 	r = np.vstack((time,grid)).T
# 	result = np.hstack((r, pro_test))
# 	## result = np.hstack((r, yest.T))
# 	## yest = yest.reshape([yest.shape[0]/20, 20])
# 	## yest = np.mean(yest, axis=1)

# 	result = pd.DataFrame(result, columns=['0','1',2,3,4,5])
# 	## result = pd.DataFrame(result, columns=['0','1',2])

# 	result = result.groupby([result['0'], result['1']]).mean()
# 	result.drop('0', axis=1, inplace=True)
# 	result.drop('1', axis=1, inplace=True)
# 	result['max'] = result.max(axis=1)
# 	result['PM'] = result.idxmax(axis=1)
# 	result = result.drop([2,3,4,5], axis=1)
# 	# result = result.reset_index()


# 	# result = result.iloc[:1011,:]
# 	# result = result.sort('max', ascending=False)
# 	result.to_csv("./Prediction/result_4_84.csv")
# 	# np.savetxt("./Prediction/yest_test.csv", yest, delimiter=",")
	


if __name__=="__main__":
    main()