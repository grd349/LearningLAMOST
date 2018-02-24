from astropy.io import fits
import matplotlib.pyplot as plt

import glob

files = glob.glob('/data2/cpb405/Training/*.fits')


for file in files:
    with fits.open(file) as hdulist:
        if hdulist[0].header['CLASS'] == 'Unknown':
            plt.plot(hdulist[0].data[0])
            plt.show()