#!/usr/bin/env python3

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
import seaborn #for plotting, to be used later


sfile = '/data2/mrs493/dr1_stellar.csv'

catalog = pd.read_csv(sfile, sep='|')

dr1 = pd.DataFrame(columns = ['designation', 'flux', 'wavelength'])

for fitsName in glob.glob('../Data/DR1/*.fits'):
    hdulist = fits.open(fitsName)
    init = hdulist[0].header['COEFF0']
    disp = hdulist[0].header['COEFF1']
    flux = hdulist[0].data[0]
    wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
    dr1.loc[len(dr1)] = [hdulist[0].header['DESIG'][7:], flux, wavelength]
    hdulist.close()

df = catalog.merge(dr1, on='designation', how='inner')

"""
if __name__ == "__main__":
    sfile = '/data2/mrs493/dr1_stellar.csv'

    df = pd.read_csv(sfile, sep='|')
    print(df.columns)
    
    print(df.loc[0:10].obsid)

    print(df.loc[df.obsid == 101005].teff)

    # Panadas make a dataframe ...

    df_spectra = pd.DataFrame(columns=['obsid', 'feature'])
    ids = [(101001, 3), (101005, 4), (101008, 10)]
    for i in ids:
        df_spectra.loc[len(df_spectra)] = [i[0], i[1]]

    print(df_spectra)
    df = df.merge(df_spectra, on='obsid', how='inner')

    print(df)

    df.to_csv('my_data.csv')
    
    fig, ax = plt.subplots()
    ax.hist(df.teff)
    plt.show()
    fig, ax = plt.subplots()
    ax.hist(df['feh'])
    plt.show()
"""