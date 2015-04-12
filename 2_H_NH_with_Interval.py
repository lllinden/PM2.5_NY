import glob
import random
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
	heating1 = data[1:2185]
	non_heating = data[2185:7225]
	heating2 = data[7225:8737]

	all = all.reshape((len(all))/(24*7), 24*7)
	heating1 = heating1.reshape((len(heating1)/(24*7),24*7))
	heating2 = heating2.reshape((len(heating2)/(24*7),24*7))
	heating = np.vstack((heating1, heating2))
	non_heating = non_heating.reshape((len(non_heating)/(24*7),24*7))


	all_avg = np.mean(all, axis=0)
	heating_avg = np.mean(heating, axis= 0)
	non_heating_avg = np.mean(non_heating, axis=0)

	all_std = np.std(all, axis=0)
	heating_std = np.std(heating, axis= 0)
	non_heating_std = np.std(non_heating, axis=0)

	n_all = all.shape[0]
	n_heating = heating.shape[0]	
	n_non_heating = non_heating.shape[0]

	t_all = stats.t.ppf(0.975,n_all)
	t_heating = stats.t.ppf(0.975,n_heating)
	t_non_heating = stats.t.ppf(0.975,n_non_heating)

	interval_all = t_all* all_std/np.sqrt(n_all)
	interval_heating = t_heating * heating_std/np.sqrt(n_heating)
	interval_non_heating  = t_non_heating * non_heating_std/(np.sqrt(n_non_heating))


	all_avg = rearrange(all_avg)
	heating_avg = rearrange(heating_avg)
	non_heating_avg = rearrange(non_heating_avg)

	interval_all = rearrange(interval_all)
	interval_heating = rearrange(interval_heating)
	interval_non_heating = rearrange(interval_non_heating)

	return all_avg, heating_avg, non_heating_avg, interval_all, interval_heating, interval_non_heating

def plot1(files):
	weekdaylst = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
	figure, axs = plt.subplots(5,2)
	plt.subplots_adjust( left=0.05, bottom=0.05, right=0.95, top=0.95)

	for i in range(2):
		count = -1
		for j in range(5):
			count += 1
			n = count + i* 5
			f = files[n]
			station_name = f[35:-8]

			heating, non_heating, heating_int, non_heating_int = reshape(f)
			heat_up = heating + heating_int
			heat_down = heating -heating_int
			non_heat_up = non_heating + non_heating_int
			non_heat_down = non_heating - non_heating_int

			axs[count,i].plot(heating, color='red')
			axs[count,i].plot(non_heating, color='green')

			# print heating_int
			# print heat_down
			axs[count,i].plot(heat_down, color='red',alpha =0.4)
			# axs[count,i].plot(heat_up, color='red',alpha =0.3)
			# axs[count,i].plot(non_heat_down, color='g',alpha =0.3)
			axs[count,i].plot(non_heat_up, color='g',alpha =0.4)


			axs[count,i].set_ylabel('pm2.5')
			axs[count,i].set_xlim([0,24*7])

			major_ticks = np.arange(0, 7*24, 24)
			minor_ticks = np.arange(0, 7*24, 1) 

			axs[count,i].tick_params(axis = 'both', which = 'major', labelsize = 12)
			axs[count,i].tick_params(axis = 'both', which = 'minor', labelsize = 12)

			axs[count,i].set_xticklabels([])

			axs[count,i].set_xticks(major_ticks)
			axs[count,i].set_xticks(minor_ticks, minor = True)
			axs[count,i].tick_params(which = 'major', direction = 'out')


			axs[count,i].grid(which = 'minor', alpha = 0.3)
			axs[count,i].grid(which = 'major', alpha = 0.9)
			axs[count,i].set_ylim([0,20])

			axs[count,i].set_title(station_name, fontsize='12' )

	axs[2,0].set_xticklabels(weekdaylst)
	axs[2,1].set_xticklabels(weekdaylst)



	plt.grid(which='major', alpha=0.5)
	

	plt.show()

def plot2(files, row):
	weekdaylst = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
	figure, axs = plt.subplots(row,2)
	plt.subplots_adjust( left=0.05, bottom=0.05, right=0.95, top=0.95)

	for i in range(2):
		count = -1
		for j in range(row):
			count += 1
			n = count + i* row
			try: 
				f = files[n]
			except IndexError:
				pass
			station_name = f[35:-8]

			all, heating, non_heating, all_int, heating_int, non_heating_int = reshape(f)

			heat_up = heating + heating_int
			heat_down = heating -heating_int
			non_heat_up = non_heating + non_heating_int
			non_heat_down = non_heating - non_heating_int

			axs[count,i].plot(heating, color='red',lw=1.5)
			axs[count,i].plot(non_heating, color='green',lw=1.5)

			axs[count, i].fill_between(range(168), heating, heat_down, color='r', alpha=0.2)
			# axs[count, i].fill_between(range(168), non_heating, non_heat_up, color='g', alpha=0.2)
			axs[count, i].fill_between(range(168), heating, heat_up, color='r', alpha=0.05)
			# axs[count, i].fill_between(range(168), non_heating, non_heat_down, color='g', alpha=0.08)


			axs[count,i].set_ylabel('pm2.5')
			axs[count,i].set_xlim([0,24*7])

			major_ticks = np.arange(0, 7*24, 24)
			minor_ticks = np.arange(0, 7*24, 1) 

			axs[count,i].tick_params(axis = 'both', which = 'major', labelsize = 12)
			axs[count,i].tick_params(axis = 'both', which = 'minor', labelsize = 12)

			axs[count,i].set_xticklabels([])

			axs[count,i].set_xticks(major_ticks)
			axs[count,i].set_xticks(minor_ticks, minor = True)
			axs[count,i].tick_params(which = 'major', direction = 'out')


			axs[count,i].grid(which = 'minor', alpha = 0.3)
			axs[count,i].grid(which = 'major', alpha = 0.9)

			axs[count,i].set_title(station_name, fontsize='12' )
			axs[count,i].set_ylim([0,20])

	axs[row-1,0].set_xticklabels(weekdaylst)
	axs[row-1,1].set_xticklabels(weekdaylst)



	plt.grid(which='major', alpha=0.5)
	

	plt.show()

if __name__ == '__main__':
	files = glob.glob('./Data/2010_ALL_STATIONS/*'+'MONITOR_'+'*.csv')
	files1 = files[4:]
	files2 = files[:4]
	# plot1(files)
	plot2(files1, 3)
	plot2(files2, 2)
	

