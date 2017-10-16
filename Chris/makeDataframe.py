#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from readFits import Spectra

if __name__ == "__main__":
    sfile = '/data2/cpb405/dr1_stellar.csv'
    spec = Spectra('../Data/DR1/*.fits')
    
    df = pd.read_csv(sfile, sep='|')
    df.drop_duplicates(subset='designation', inplace=True)

    #print(df.columns)

    #print(df.loc[0:10].obsid)

    #print(df.loc[df.obsid == 101005].teff)

    # Panadas make a dataframe ...
    """
    df_spectra = pd.DataFrame(columns=['obsid', 'feature'])
    ids = [(101001, 3), (101005, 4), (101008, 10)]
    for i in ids:
        df_spectra.loc[len(df_spectra)] = [i[0], i[1]]
    """
    #print(spec.colourList)
    df_spectra = pd.DataFrame(columns=['designation', 'feature', 'wavelength', 'flux'])
    ids = np.column_stack((spec.desigList, spec.colourList, spec.wavelengthList, spec.fluxList))
    for i in ids:
        df_spectra.loc[len(df_spectra)] = [i[0], i[1], i[2], i[3]]

    df = df.merge(df_spectra, on='designation', how='inner')
    
    fig, ax = plt.subplots()
    ax.scatter([df.feature], df.teff)
    ax.set_xlabel('log(B/V) Feature')
    ax.set_ylabel('Effective Temperature / K')
    ax.set_title('Plot of Temperature of Stars Against Colour')
    plt.savefig("Tempcolplot")
    
    df.to_csv('my_data.csv')
    
    """
    fig, ax = plt.subplots()
    ax.hist(df.teff)
    plt.show()
    fig, ax = plt.subplots()
    ax.hist(df['feh'])
    plt.show()
    """
    