#!/usr/bin/env python3

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

sfile = '/data2/mrs493/dr1_stellar.csv'

catalog = pd.read_csv(sfile, sep='|')

catalog.drop_duplicates(subset = 'designation', inplace = True)

dr1 = pd.DataFrame(columns = ['designation', 'totalCounts', 'B', 'V', 'R', 'colour', 'filename'])

letters = {'B':[3980, 4920], 'V':[5070, 5950], 'R':[5890, 7270]}

for fitsName in glob.glob('/data2/mrs493/DR1/*.fits'):
    hdulist = fits.open(fitsName)
            
    init = hdulist[0].header['COEFF0']
    disp = hdulist[0].header['COEFF1']
    flux = hdulist[0].data[0]
    
    wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
    totalCounts = sp.sum(flux)

    bands = {'B':0, 'V':0, 'R':0}
            
    for letter in letters:
        lower = sp.searchsorted(wavelength, letters[letter][0], side = 'left')
        upper = sp.searchsorted(wavelength, letters[letter][1], side = 'right')
        #bands[letter] = -2.5*sp.log10(sp.sum(flux[lower:upper]))
        bandFlux = flux[lower:upper]
        bandFlux[bandFlux<0] = sp.nan
        bands[letter] = -2.5*sp.log10(sp.nanmean(bandFlux))
                
    colour = bands['B'] - bands['V']
    
    dr1.loc[len(dr1)] = [hdulist[0].header['DESIG'][7:], totalCounts, bands['B'], bands['V'], bands['R'], colour, hdulist[0].header['FILENAME']]
        #why have to take real parts for log? due to logging 0 somewhere?
    
    hdulist.close()

df = catalog.merge(dr1, on='designation', how='inner')

df.to_csv('/data2/mrs493/my_data2.csv')