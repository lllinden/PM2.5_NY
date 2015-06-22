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

np.random.seed(333)

dat = pd.read_csv('./Spatial_Data/sim_4_1.csv')
# print dat.head()
X = dat.iloc[:,1:]
print X.iloc[:,0].unique()
# print kk

print X.head()
input_f = X.shape[1]-2

data = np.random.permutation(X.values)
data[:,5:] = data[:,5:] / data[:,5:].max(axis=0)

trndata = ClassificationDataSet(input_f,1)
tstdata = ClassificationDataSet(input_f,1)

spilt = int(data.shape[0]*0.7)
for i in range(spilt):
	indata = data[i,2:]
	outdata = data[i,1]
	trndata.addSample(indata,outdata)

tst_id =[]
for i in range(spilt, data.shape[0]):
	tst_id.append(data[i,0])
	indata = data[i,2:]
	outdata = data[i,1]
	tstdata.addSample(indata,outdata)

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

fnn = buildNetwork(trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(fnn, learningrate = 0.001, dataset=trndata,momentum=0.01, verbose=True, weightdecay=0.01)

for i in range(5000):
	trainer.trainEpochs(1)
	trnresult = percentError(trainer.testOnClassData(),trndata['class'])
	tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

	print "epoch: %4d" % trainer.totalepochs, \
		  "  train error: %5.2f%%" % trnresult, \
		  "  test error: %5.2f%%" % tstresult

print tst_id
print len(tst_id)
print trainer.testOnClassData(dataset=tstdata)
# print tstdata['class']
