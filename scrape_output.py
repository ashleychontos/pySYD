"""
Grabs each individual target's results and concatenates results into a single csv in Files/ for each submodulel
(i.e. findex.csv and globalpars.csv). This is automatically called at the end of the main SYD module.
"""

import glob
import pandas as pd


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
    df_new.set_index('parameter', inplace=True, drop=False)
    df.loc[i, 'target'] = file.strip().split('/')[-2]
    for col in df_new.index.values.tolist():
        df.loc[i, col] = df_new.loc[col, 'value']

df.fillna('--', inplace=True)
df.to_csv('Files/globalpars.csv', index=False)
