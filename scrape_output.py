import glob
import pandas as pd

def _csv_reader(f):
    row = pd.read_csv(f,header=None,squeeze=True, index_col=0)
    return row

path = 'Files/results/**/*findex.csv'
fL = glob.glob(path)
df = pd.read_csv(fL[0])

for i in range(1,len(fL)):
	   dfnew = pd.read_csv(fL[i])
	   df = pd.concat([df,dfnew])

df.to_csv('Files/findex.csv',index=False)

path = 'Files/results/**/*globalpars.csv'
files = glob.glob(path)

df = pd.DataFrame(columns=['target'])

for i, file in enumerate(files):
    df_new = pd.read_csv(file)
    df_new.set_index('parameter',inplace=True,drop=False)
    df.loc[i,'target']=file.strip().split('/')[-2]
    for col in df_new.index.values.tolist():
        df.loc[i,col]=df_new.loc[col,'value']

df.fillna('--',inplace=True)
df.to_csv('Files/globalpars.csv',index=False)