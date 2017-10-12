#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sfile = 'Data/dr3_stellar.csv'

    df = pd.read_csv(sfile, sep='|')
    print(df.columns)

    fig, ax = plt.subplots()
    ax.hist(df.teff)
    plt.show()
    fig, ax = plt.subplots()
    ax.hist(df['feh'])
    plt.show()
