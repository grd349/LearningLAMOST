import pandas as pd
from astropy.io import fits
import glob

import time

print('Opening catalog...')
t = time.time()
cfile = '/data2/cpb405/dr1.csv'
catalog = pd.read_csv(cfile, sep='|')
catalog.drop_duplicates(subset = 'designation', inplace = True)
print('Catalog opened: 'time.time() - t, '\nReading in FITS files...')

sfile = '/data2/mrs493/DR1_3/*.fits'
files = glob.glob(sfile)

keys = ['designation', 'filename']

dr1 = pd.DataFrame(columns=keys)
t = time.time()
for fitsName in files:
    with fits.open(fitsName) as hdulist:    
        df = pd.DataFrame(columns=keys)
        df.loc[0] = [hdulist[0].header['DESIG'][7:], hdulist[0].header['FILENAME']]
        dr1 = pd.concat([dr1, df])
print('FITS files read: 'time.time() - t, '\nMerging DataFrames')

df = catalog.merge(dr1, on='designation', how='inner')

classification = df['class'].values
STAR = [classification == 'STAR']
classification[STAR] = [mk[0] for mk in df['subclass'].values[STAR]]
df['classification'] = classification

file = 'classification.csv'

df[['filename', 'classification']].to_csv(file, index = False)
print('DataFrames Merged\nResult saved to ', file)