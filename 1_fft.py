from pandas import read_csv
from scipy import stats
import sys, datetime, pandas, numpy, scipy, math 
import matplotlib.pyplot as plt

try:
	station = sys.argv[1]
except IndexError:
	print 'please choose a station: 11MONITOR_CCNY, 12MONITOR_DS, 13MONITOR_IS74, 14MONITOR_PS19, 21MONITOR_PS274, 22MONITOR_IS314, 31MONITOR_IS143, 41MONITOR_MASPETH, 51MONITOR_FKW, 61MONITOR_Average'

def readCSV(station):
	data = read_csv('./Data/2010_ALL_STATIONS/'+ station + '2010.csv')
	return data
def timeSeriesDF(data):
	t = pandas.date_range('2010-01-01 00:00', '2010-12-31 23:00', freq='h')
	dataTS  = data.set_index(t)
	# dataTS = dataTS.drop('Unnamed: 0', 1)
	dataTS = dataTS.drop('Date', 1)
	dataTS = dataTS.drop('Time', 1)
	# dataTS = dataTS.drop('index1',1)
	return dataTS
def fft(ts):
	data_fft = ts.iloc[0:record_number,:]
	print data_fft
	pm =  data_fft['PM25C'].values
	avg = numpy.mean(pm)
	pm_cen = []
	for i in pm:
		x = i -avg
		pm_cen.append(x)
	pm_fft = numpy.fft.fft(pm_cen)
	freqs = numpy.fft.fftfreq(len(pm_cen))	
	n = len(pm_cen)
	p = []
	for i in pm_fft:
		x = numpy.real(i)**2 + numpy.imag(i)**2
		p.append(x/n)
	return p, freqs
def get_summit(p,f,n):
	p1 = p[0:len(p)/2]
	f1 = f[0:len(freqs)/2]
	p_sort = sorted(zip(p1,f1), key = lambda elm:elm[0], reverse=True)
	ssum = sum(p)
	test = test_summit(p_sort, ssum, n, record_number)
	summit = []
	for x in test:
		print x
		summit.append(x[-1]*record_number/24)
	return summit
def test_summit(p_sort,ssum,n, N):
	p_sort = p_sort[0:len(p_sort)/2]
	test = []
	i = 0
	for j in range(n+1):
		i += 1
		t = p_sort[j][0]/ssum
		# f = stats.distributions.f.ppf(0.95, i, 8192)
		if t > 0.0009:
			test.append(p_sort[j])
	return test

def plot(x,y,n):
	p1 = p[0:len(p)/n]
	freqs1 = freqs[0:len(freqs)/n]
	plt.plot(freqs1, p1, color = 'indigo')
	plt.xlabel('frequency')
	plt.ylabel('magnitude of frequency')
	plt.title('FFT of PM2.5, 2010 '+station)
	plt.show()

data = readCSV(station)
dataTS = timeSeriesDF(data)
dataTS = dataTS.fillna(method='pad')
record_number = 2**int(math.log(len(dataTS),2))

p, freqs = fft(dataTS)
summits = get_summit(p,freqs, 10)
for summit in summits:	
	print '%.2f'%summit
# plot(freqs, p, 2)
plot(freqs , p, 10)
plt.xlabel('frequency')
plt.ylabel('magnitude of frequency')
plt.title('Period analysis of PM2.5, 2010 '+station)
