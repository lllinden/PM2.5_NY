import os 
import glob
import pandas as pd
# data = glob.glob('./2010_ALL_STATIONS/'+'*.csv')
# df = pd.read_csv(data[8])
# print data[8]

data = './STATIONS/MONITOR_PS12.csv'
df = pd.read_csv(data)

# print len(df)
t = df.loc[:,['Date','Time']].values.tolist()
# print t
t1 = []
for i in range(len(t)):
	try:
		t1.append(int(t[i][-1][:2]))
	except ValueError:
		t1.append(int(t[i][-1][:1]))

for i in range(365):
	hour = 0
	for j in range(24):
		index = i*24+j
		if t1[index] != hour:
			# print t1[index]
			# print hour
			print i,j
			break
		else:
			hour+=1
