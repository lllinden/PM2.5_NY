import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, random, glob, sys

files = glob.glob('./Turnstile/*.csv')
abnormal_list = []
for f in files:
	# f = './Turnstile/.csv'
	with open(f, 'rb') as csvfile:
	    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	    data = [row for row in spamreader]

	### the list of control area in the data
	control_area = set([d[0] for d in data[:]])
	### the list of remote station in the data
	remonte = set([d[1] for d in data[:]])
	# print len(control_area)
	# print len(remonte)


	with open('Remote-Booth-Station.csv', 'rU') as csvfile:
	    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	    stations = [row for row in spamreader]
	# The station under remote dictionary
	remoteToStations = {x[0]:[x[2],x[3]] for x in stations[1:]}
	# print remoteToStations

	station = sys.argv[1]
	oneStation = [d for d in data if d[1]==station]
	### contains the key for one station
	X = {}
	for x in oneStation:
	    if x[2] not in X.keys():
	        X[x[2]] = x[3:]
	    else:
	        X[x[2]] += x[3:]

	k = X.keys()[0]


	# # Interpolation
	# 
	# * Notice **Monday February 17, 2014** was the **President's Day** Holiday

	scp = X.keys()
	scpDict = dict((key,[]) for key in scp)
	scpDict1 = dict((key,[]) for key in scp)
	# print scpDict

	# print oneStation
	for d in oneStation:
		d[-1] = d[-1][:9] 
		Y = []
		Y += d[3:]
	# # each time slot has 5 columns : Date, Time, Classification, Entry, Exit    
		for t in range(len(Y)/5):
			scpDict[d[2]].append([Y[5*t + n] for n in range(5)])
	# print scpDict


	for x in X.keys():
		scp_df = pd.DataFrame(scpDict[x])
	    ### gave df column name
		scp_df.columns = ['Date', 'Time','Type','Entry','Exit']
		##split the time into hour and min and use hour to replace the time    
		scp_df1 = pd.DataFrame(scp_df['Time'].str.split(':',1).tolist(), columns=['H','Ms'])
		scp_df['Time'] = scp_df1['H'] 
		### Drop the duplicate records 
		scp_df = scp_df[scp_df['Type'] != 'RECOVR']

	    
	    ### set two column as double index
		scp_df = scp_df.set_index(['Date', 'Time'])
	    ### calculate the difference of entry and exit in every hour
		scp_entry_arr = scp_df['Entry'].values
		scp_entry_arr = scp_entry_arr.astype(np.float)
		scp_entry_diff = scp_entry_arr[1:]-scp_entry_arr[:-1]
		scp_entry_diff = np.append(scp_entry_diff, 0)

		scp_exit_arr = scp_df['Exit'].values
		scp_exit_arr = scp_exit_arr.astype(np.float)
		scp_exit_diff = scp_exit_arr[1:]-scp_exit_arr[:-1]
		scp_exit_diff = np.append(scp_exit_diff, 0)
	    
		scp_df['Total_entry']= scp_entry_diff
		scp_df['Total_exit']= scp_exit_diff

		scpDict1[x] = scp_df

	c = 0
	time_lst = []
	for x in scp[1:]:
		entry = scpDict1[x].iloc[:,3:4]
		exit = scpDict1[x].iloc[:,4:]
		# entry.dropna(inplace='TRUE')
		# exit.dropna(inplace='TRUE')
		# exit.fillna(0)
		# entry.fillna(0)
		col5 = 'Total_entry_' + x
		col6 = 'Total_exit_' + x
		entry.columns = [col5]
		exit.columns = [col6]


		### delet the duplicate index and keep the last one 
		group_entry = entry.groupby(level = entry.index.names)
		group_exit = exit.groupby(level = exit.index.names)
		entry = group_entry.last()
		exit = group_exit.last()
		### list all the time
		for x in entry.index:
			if x[-1] not in time_lst:
				time_lst.append(x[-1])

		if c == 0:
			entry_df = entry
			exit_df =exit
			if len(entry.index) == 42:
				print 'haha'
				index = entry.index
			else:
				# print len(entry.index)
				print f
				abnormal_list.append(f)
				# print entry.index
			c = 1
		else:
			# if len(entry.index) < 42:
			# 	print len(entry),'entry'
			# if len(exit.index) < 42:
			# 	print len(exit), 'exit'
			# print exit

			entry_df = entry_df.join(entry, how ='left')
			exit_df = exit_df.join(exit, how = 'left')

	entry.sum(axis=1).to_csv('./Entry/' + f[12:])
	exit.sum(axis=1).to_csv('./Exit/' + f[12:])

for f in abnormal_list:
	f1 = './Entry/' + f[-10:]
	f2 = './Exit/' + f[-10:]
	for ff in [f1,f2]:
		with open(ff, 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
			data = [row for row in spamreader]
			data1 = [data[0]]
			t0 = int(data[0][1])
			c = 0
			for x in data[1:]:
				c += 1
				if int(x[1]) == t0 + (c%6) *4: 
					data1.append(x)
				else:
					c -= 1
			with open(ff, 'wb') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerows(data1)

files_Entry = glob.glob('./Entry/*.csv')
files_Exit = glob.glob('./Exit/*.csv')

for f in [files_Entry, files_Exit]:
	c = 0
	data1=[]
	for ff in f:
		with open(ff, 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
			data = [row for row in spamreader]
		for x in data:
			data1.append(x)

	file = './Output/' + ff[1:6] + station +'.csv'
	with open(file, 'wb') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerows(data1)

