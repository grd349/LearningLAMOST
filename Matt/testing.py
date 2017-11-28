#!/usr/bin/env python3

import pandas as pd
import scipy as sp
import glob
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from fits import Spectrum
'''
sfile = '/data2/mrs493/my_data2.csv'

df = pd.read_csv(sfile, sep=',')

fig, ax = plt.subplots()
ax.scatter(df.teff.tolist(), df.Ha.tolist())
ax.set_xlabel('Effective temperature \ K')
ax.set_ylabel('Ha Equivalent Width \ Angstroms')
ax.set_title('Effective Temperature vs. Ha Equivalen Width')

plt.show()
'''

classes = {}

for fitsName in glob.glob('/data2/mrs493/DR1_3/*.fits'):
    hdulist = fits.open(fitsName)
    CLASS = hdulist[0].header['CLASS']
    try:
        classes[CLASS] += 1
    except:
        classes[CLASS] = 1

print classes
        