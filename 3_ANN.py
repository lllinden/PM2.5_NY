import pandas as pd
import numpy as np
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

# np.random.seed(333)

dat = pd.read_csv('./Spatial_Data/PM2.5_Spatial_0.csv')
X = dat.iloc[:,2:]
input_f = X.shape[1]-2

for kk in range(10):
	data = np.random.permutation(X.values)
	data[:,:-1] = data[:,:-1] / data[:,:-1].max(axis=0)

	trndata = ClassificationDataSet(input_f,1)
	tstdata = ClassificationDataSet(input_f,1)

	spilt = int(data.shape[0]*0.7)
	for i in range(spilt):
		indata = data[i,1:-1]
		outdata = data[i,-1]
		trndata.addSample(indata,outdata)

	for i in range(spilt, data.shape[0]):
		indata = data[i,1:-1]
		outdata = data[i,-1]
		tstdata.addSample(indata,outdata)

	trndata._convertToOneOfMany( )
	tstdata._convertToOneOfMany( )

	fnn = buildNetwork(trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer)
	trainer = BackpropTrainer(fnn, learningrate = 0.01, dataset=trndata,momentum=0.01, verbose=True, weightdecay=0.01)

	for i in range(300):
		trainer.trainEpochs(1)
		trnresult = percentError(trainer.testOnClassData(),trndata['class'])
		tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

		print "epoch: %4d" % trainer.totalepochs, \
			  "  train error: %5.2f%%" % trnresult, \
			  "  test error: %5.2f%%" % tstresult

	print tstdata['class']
	print trainer.testOnClassData(dataset=tstdata)
