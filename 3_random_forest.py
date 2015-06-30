from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


print 'working'

def main():
    #create the training & test sets, skipping the header row with [1:]

	np.random.seed(333)
	dat = pd.read_csv('./Spatial_Data/test_corr_1000.csv')
	X = dat.iloc[:,4:]

	data = np.random.permutation(X.values)
	data[:,4:-1] = data[:,4:-1] / data[:,4:-1].max(axis=0)
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
	yest = rf.predict(testX)
	pro = rf.predict_proba(testX)
	# for x in pro:
		# print x

if __name__=="__main__":
    main()