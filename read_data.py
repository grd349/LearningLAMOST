#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

if __name__ == "__main__":
    sfile = '/data2/dr3_stellar.csv'

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
