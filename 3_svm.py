import random, timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import cross_validation

random.seed(333)

dat = pd.read_csv('./Data/PM2.5_Weather_3Cat_3.csv')
print 'Features:', dat.columns[2:]
X = dat.iloc[:,2:].values
y = dat['PM2.5'].values
y = y.reshape((X.shape[0], 1))

dat_random = np.concatenate((X, y), axis=1)
train = np.random.permutation(dat_random)
X_random = train[:,:-1]
y_random = train[:,-1]

a = int(X.shape[0])*0.6

Xtrain = X_random[:a,:]
Xtest = X_random[a:a*1.5,:]
ytrain = y_random[:a]
ytest = y_random[a:a*1.5]

print '# of training samples', int(a)
start = timeit.default_timer()
# clf = svm.SVC(kernel='rbf', cache_size = 40, gamma = 0.0, C = 2.5, random_state=333)
clf = svm.SVC(kernel='rbf', cache_size = 40, gamma = 0.065, C = 2.8, random_state=333)
# print 'rbf model', clf
model = clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
ypred1 = clf.predict(Xtrain)

# scores = cross_validation.cross_val_score(clf, Xtrain, ytrain, cv=5)
stop = timeit.default_timer()
print 'building model rbf', stop-start
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
acc = accuracy_score(ytest,ypred)
acc1 = accuracy_score(ytrain,ypred1)
print acc
print acc1
