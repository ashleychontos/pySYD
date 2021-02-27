"""
Grabs each individual target's results and concatenates results into a single csv in Files/ for each submodulel
(i.e. findex.csv and globalpars.csv). This is automatically called at the end of the main SYD module.
"""

import glob
import pandas as pd
from itertools import chain


# TODO: This doesn't seem to be used
def _csv_reader(f):
    row = pd.read_csv(f, header=None, squeeze=True, index_col=0)
    return row


# Concatenate find excess results
path = 'Files/results/**/*findex.csv'
fL = glob.glob(path)
df = pd.read_csv(fL[0])

for i in range(1, len(fL)):
    dfnew = pd.read_csv(fL[i])
    df = pd.concat([df, dfnew])

df.to_csv('Files/findex.csv', index=False)

# Concatenate fit background results
path = 'Files/results/**/*globalpars.csv'
files = glob.glob(path)

df = pd.DataFrame(columns=['target'])

for i, file in enumerate(files):
	df_new = pd.read_csv(file)
	df_new.set_index('parameter',inplace=True,drop=False)
	df.loc[i,'target']=file.strip().split('/')[-2]
	new_header_names=[[i,i+'_err'] for i in df_new.index.values.tolist()] #add columns to get error
	new_header_names=list(chain.from_iterable(new_header_names))          
	for col in new_header_names:
		if '_err' in col:
			df.loc[i,col]=df_new.loc[col[:-4],'uncertainty']
		else:
			df.loc[i,col]=df_new.loc[col,'value']

df.fillna('--', inplace=True)
df.to_csv('Files/globalpars.csv', index=False)
