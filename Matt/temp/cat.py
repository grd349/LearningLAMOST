import pandas as pd
import glob

[pd.read_csv(x, sep = ',') for x in glob.glob('files/*.csv')]

df = pd.concat([pd.read_csv(x, sep = ',') for x in glob.glob('files/*.csv')])

df.to_csv('spectra.csv')

