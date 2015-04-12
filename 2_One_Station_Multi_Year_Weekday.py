import glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy import genfromtxt
from scipy import signal

def rearrange(average, year):
	if year == 2010:
		avg = np.hstack((average[72:],average[:72]))
	if year == 2011:
		avg = np.hstack((average[48:],average[:48]))
	if year == 2012:
		avg = np.hstack((average[24:],average[:24]))
		print average
	if year == 2013:
		avg = np.hstack((average[144:],average[:144]))
	if year == 2014:
		avg = np.hstack((average[120:],average[:120]))
	return avg

def reshape(f, year):
	data = genfromtxt(f, delimiter=',')[:, -1]
	all = data[1:8737]
	all = all.reshape((len(all))/(24*7), 24*7)
	all_avg = np.mean(all, axis=0)
	all_avg = rearrange(all_avg, year)
	return all_avg


def plot(files):
	weekdaylst = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
	colors = ['FIREBRICK','CRIMSON','DARKOLIVEGREEN','DARKGREEN','YELLOWGREEN']
	fig, ax2 = plt.subplots(1,1)
	plt.subplots_adjust( left=0.05, bottom=0.05, right=0.95, top=0.95)

	for count in range(5):
		f = files[count]
		year = int(f[-8:-4])
		all = reshape(f, year)
		ax2.plot(all, color = colors[count], label = year)
		# ax2.plot(all)
		ax2.set_ylabel('pm2.5')
		ax2.set_xlim([0,24*7])
		ax2.set_ylim([5,17])

		major_ticks = np.arange(0, 7*24, 24)
		minor_ticks = np.arange(0, 7*24, 1) 

		ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
		ax2.tick_params(axis = 'both', which = 'minor', labelsize = 12)

		ax2.set_xticklabels([])

		ax2.set_xticks(major_ticks)
		ax2.set_xticks(minor_ticks, minor = True)
		ax2.tick_params(which = 'major', direction = 'out')


		ax2.grid(which = 'minor', alpha = 0.3)
		ax2.grid(which = 'major', alpha = 0.9)

		ax2.set_title(year, fontsize='12' )
	ax2.set_xticklabels(weekdaylst)

	handles, labels = ax2.get_legend_handles_labels()
	ax2.legend(handles, labels)

	plt.grid(which='major', alpha=0.5)
	

	plt.show()

if __name__ == '__main__':
	files = glob.glob('./STATIONS/MONITOR_'+'*.csv')
	plot(files)
	

