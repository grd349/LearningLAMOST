import pandas as pd
import glob

df = pd.concat([pd.read_csv(x, sep = ',') for x in glob.glob('*.csv')])

df.to_csv('../spectra.csv')

