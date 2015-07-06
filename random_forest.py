from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from scipy import stats
import sys

print 'working'

def main():
    #create the training & test sets, skipping the header row with [1:]
	n_month = sys.argv[1]
	np.random.seed(33)
	train_0 = pd.read_csv('./Spatial_Data/Self_Training/train_'+n_month+'.csv')
	test_0 = pd.read_csv('./Spatial_Data/Self_Training/test_'+n_month+'.csv')
	predict_0 = pd.read_csv('./Prediction/data_trans_20_new.csv')
	# print predict_0.iloc[:10,:]

	train = train_0.iloc[:,4:].values
	test = test_0.iloc[:,5:].values
	predict = predict_0.iloc[:,5:].values

	train_random = np.random.permutation(train)
	train_random[:,4:-1] = train_random[:,4:-1] / train_random[:,4:-1].max(axis=0)
	test[:,4:-1] = test[:,4:-1] / test[:,4:-1].max(axis=0)
	predict[:,4:-1] = predict[:,4:-1] / predict[:,4:-1].max(axis=0)

	trainY = train_random[:,-1]
	trainX = train_random[:,:-1]
	testY = test[:,-1]
	testX = test[:,:-1]

	rf = RandomForestClassifier(n_estimators=200,max_depth=4,oob_score=True)
	rf.fit(trainX, trainY)
	acc = rf.score(testX, testY)
	print acc
	# print rf.feature_importances_
	# yest = rf.predict(testX)
	# pro = rf.predict_proba(testX)

	# yest = yest.reshape(([yest.shape[0]/10,10]))
	# yest = stats.mode(yest.T)[0]
	# testY = testY.reshape(([testY.shape[0]/10,10]))
	# testY = stats.mode(testY.T)[0]

	# count = 0
	# result = np.vstack((yest, testY)).T
	# for i in range(result.shape[0]):
	# 	if result[i,0] == result[i,1]:
	# 		count +=1
	# print count/float(result.shape[0])
	# yest = rf.predict(predict)
	# yest = yest.reshape(([yest.shape[0]/10,10]))
	# yest = stats.mode(yest.T)[0].T

	# time = predict_0.iloc[:,0].values
	# grid = predict_0.iloc[:,1].values
	yest = rf.predict(predict)
	# pro_test = rf.predict_proba(predict)
	# r = np.vstack((time,grid)).T
	# result = np.hstack((r, pro_test))
	# result = np.hstack((r, yest.T))
	yest = yest.reshape([yest.shape[0]/20, 20])
	yest = np.mean(yest, axis=1)

	# result = pd.DataFrame(result, columns=['0','1',2,3,4])
	# result = pd.DataFrame(result, columns=['0','1',2])

	# result = result.groupby([result['0'], result['1']]).mean()
	# result.drop(0, axis=1, inplace=True)
	# result.drop(1, axis=1, inplace=True)
	# result['max'] = result.max(axis=1)
	# result['PM'] = result.idxmax(axis=1)
	# result = result.drop([2,3,4], axis=1)
	# result = result.reset_index()


	# result = result.iloc[:1011,:]
	# result = result.sort('max', ascending=False)
	# result.to_csv("./Prediction/test_nu_11.csv")
	np.savetxt("./Prediction/yest_test.csv", yest, delimiter=",")
	


if __name__=="__main__":
    main()