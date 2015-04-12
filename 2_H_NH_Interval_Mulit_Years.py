import sys, glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy import genfromtxt
from scipy import signal, stats


def rearrange(average, year):
	if year == 2010:
		avg = np.hstack((average[72:],average[:72]))
	if year == 2011:
		avg = np.hstack((average[48:],average[:48]))
	if year == 2012:
		avg = np.hstack((average[24:],average[:24]))
	if year == 2013:
		avg = np.hstack((average[144:],average[:144]))
	if year == 2014:
		avg = np.hstack((average[120:],average[:120]))
	return avg

def reshape(f, year):
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


	all_avg = rearrange(all_avg, year)
	heating_avg = rearrange(heating_avg, year)
	non_heating_avg = rearrange(non_heating_avg, year)

	interval_all = rearrange(interval_all, year)
	interval_heating = rearrange(interval_heating, year)
	interval_non_heating = rearrange(interval_non_heating, year)

	return  heating_avg, non_heating_avg, interval_heating, interval_non_heating

def plot(files):
	weekdaylst = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
	figure, axs = plt.subplots(5,1)
	plt.subplots_adjust( left=0.05, bottom=0.05, right=0.95, top=0.95)

	for count in range(5):
		f = files[count]
		year = int(f[-8:-4])

		heating, non_heating, heating_int, non_heating_int = reshape(f, year)
		heat_up = heating + heating_int
		heat_down = heating -heating_int
		non_heat_up = non_heating + non_heating_int
		non_heat_down = non_heating - non_heating_int

		axs[count].plot(heating, color='red')
		axs[count].plot(non_heating, color='green')
		axs[count].fill_between(range(168), heating, heat_down, color='r', alpha=0.2)
		axs[count].fill_between(range(168), heating, heat_up, color='r', alpha=0.04)

		axs[count].set_xlim([0,24*7])
		axs[count].set_ylim([0,25])

		major_ticks = np.arange(0, 7*24, 24)
		minor_ticks = np.arange(0, 7*24, 1) 

		axs[count].tick_params(axis = 'both', which = 'major', labelsize = 12)
		axs[count].tick_params(axis = 'both', which = 'minor', labelsize = 12)

		axs[count].set_xticklabels([])

		axs[count].set_xticks(major_ticks)
		axs[count].set_xticks(minor_ticks, minor = True)
		axs[count].tick_params(which = 'major', direction = 'out')


		axs[count].grid(which = 'minor', alpha = 0.3)
		axs[count].grid(which = 'major', alpha = 0.9)
		axs[count].set_title(year, fontsize='12')


	axs[-1].set_xticklabels(weekdaylst)

	plt.grid(which='major', alpha=0.5)
	plt.show()


if __name__ == '__main__':
	files = glob.glob('./Data/PS19_2010_2014/'+'*.csv')
	# files = glob.glob('./STATIONS/MONITOR_'+'*.csv')
	plot(files)
	
