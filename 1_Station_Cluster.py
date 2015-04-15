from sklearn import cluster, datasets
import sys, glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy import genfromtxt
from scipy import signal, stats


def rearrange(average):
	avg = np.hstack((average[72:],average[:72]))
	return avg

def reshape(f):
	data = genfromtxt(f, delimiter=',')[:, -1]
	all = data[1:8737]
	all = all.reshape((len(all))/(24*7), 24*7)
	all_avg = np.mean(all, axis=0)
	all_avg = rearrange(all_avg)

	weekend = all_avg[120:]
	weekday = all_avg[:120]
	weekday = weekday.reshape((5,24))
	weekday_avg = np.mean(weekday, axis = 0)

	week = np.append(weekday_avg,weekend)
	return all_avg, week

def plot(result, files, cluster):
	c = 0
	for i in range(cluster):
		if c == 0:
			plot_dct = {i:[]}
			c = 1
		else:
			plot_dct[i] = []

	for i in range(len(files)):
		all, data_km = reshape(files[i])
		cl = result[i]
		plot_dct[cl].append(all)

	weekdaylst = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
	figure, axs = plt.subplots(cluster,1)
	plt.subplots_adjust( left=0.05, bottom=0.05, right=0.95, top=0.95)

	for i in range(cluster):
		plot_dct[i] = np.mean(np.vstack(plot_dct[i]),axis=0)
		if i == 0:
			axs[i].plot(plot_dct[i], color='green',lw=1.5)
		if i == 1:
			axs[i].plot(plot_dct[i], color='red',lw=1.5)
		if i == 2:
			axs[i].plot(plot_dct[i], color='blue',lw=1.5)

		axs[i].set_ylabel('pm2.5')
		axs[i].set_xlim([0,24*7])

		major_ticks = np.arange(0, 7*24, 24)
		minor_ticks = np.arange(0, 7*24, 1) 

		axs[i].tick_params(axis = 'both', which = 'major', labelsize = 12)
		axs[i].tick_params(axis = 'both', which = 'minor', labelsize = 12)

		axs[i].set_xticklabels([])

		axs[i].set_xticks(major_ticks)
		axs[i].set_xticks(minor_ticks, minor = True)
		axs[i].tick_params(which = 'major', direction = 'out')


		axs[i].grid(which = 'minor', alpha = 0.3)
		axs[i].grid(which = 'major', alpha = 0.9)

		# axs[i].set_title(station_name, fontsize='12' )
		axs[i].set_ylim([5,15])

	axs[-1].set_xticklabels(weekdaylst)

	plt.grid(which='major', alpha=0.5)
	plt.show()

if __name__ == '__main__':
	files = glob.glob('./Data/2010_ALL_STATIONS/*'+'2010.csv')
	# print files
	c = 0
	for f in files:
		if c == 0 :
			all, data_km = reshape(f)
			c = 1
		else:
			all, week = reshape(f)
			data_km = np.vstack((data_km,week))

	try:
		clusters = sys.argv[1]
	except IndexError:
		print 'please tell me how many clusters for k-mean algorithm, 2 or 3'

	
	for i in range(500):
		km = cluster.KMeans(n_clusters=int(sys.argv[1]))
		km.fit(data_km) 
		centers = km.cluster_centers_
		centers = centers.round(2)
		prediction = km.predict(data_km)	
		if c == 1:
			result = prediction
			c = 2
		else:
			result = np.vstack((result, prediction))

	result = np.mean(result, axis=0)
	result[result<=0.5] = 0
	result[(result>0.5) & (result<1)] = 1
	result[result>1] = 2

	station_list = ['CCNY','DS', 'IS74', 'PS19', 'PS274', 'PS314', 'IS143', 'MASPETH','FKW']
	for i in range(len(station_list)):
		print station_list[i], ': ', result[i] 

	plot_dct = plot(result, files, int(sys.argv[1]))
	print plot_dct
