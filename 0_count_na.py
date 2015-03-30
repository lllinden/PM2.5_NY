import glob
import pandas as pd

files = glob.glob('./2010_ALL_STATIONS/'+'*.csv')

non_na = []
print files
for x in files:
	df = pd.read_csv(x)
	non_na_count = df.count()
	non_na.append(non_na_count)

################
## CCNY: 8760
## DS: 8759
## FKW: 8635
## IS143: 8759
## IS74: 8759
## MASPETH: 8759
## PS19: 8759
## PS274: 8759
## PS314: 8759
################

