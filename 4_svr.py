import random, timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import cross_validation

from sklearn.grid_search import GridSearchCV



dat = pd.read_csv('./Temporal_Data/All_station_2014_nu_temp.csv')
dat_arr = dat.values
b =0.3 
a = dat_arr.shape[0] * b
train = dat_arr[:a,:]
test = dat_arr[a:,:]

train = np.random.permutation(train)
Xtrain = train[:,1:]
ytrain = train[:,0]
Xtest = test[:,1:]
ytest = test[:,0]

# print '# of training samples', int(a)
start = timeit.default_timer()
lst = []
for g in [0.1]:
	for c in [0.1,0.5,1,5,10,50,100]:
		random.seed(333)
		np.random.seed(333)
#C_range = 10.0 ** np.arange(-4, 4)
#gamma_range = 10.0 ** np.arange(-4, 4)
#param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())
#svr = svm.SVR()
#grid = GridSearchCV(svr, param_grid)
# grid.fit(Xtrain, ytrain)
# print("The best classifier is: ", grid.best_estimator_)
# print(grid.grid_scores_)
		clf = svm.SVR(kernel='rbf', cache_size = 40, gamma = g, C = c)
		print 'rbf model', clf
		model = clf.fit(Xtrain, ytrain)
		r2 = clf.score(Xtest, ytest)
# 		yest = clf.predict(Xtest)
# 		result = pd.DataFrame()
# 		result['yest'] = yest
# 		result['y'] = ytest
# #lst.append(r2)
		print r2
#r2_record = np.array(lst)
#r2_record = r2_record.reshape((9,6))
#np.savetxt('r2_svr_all.csv', r2_record, delimiter=',')

# result.to_csv('result.csv')

# scores = cross_validation.cross_val_score(clf, Xtrain, ytrain, cv=5)
stop = timeit.default_timer()
print 'building model rbf', stop-start
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
