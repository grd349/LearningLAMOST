from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

import glob

#33623 - looks like star

#i = [1493, 1545, 3445, 26168]
i = [1545, 26168]

files = [glob.glob('/data2/cpb405/Training_2/*.fits')[j] for j in i]
#files = glob.glob('/data2/cpb405/Training_2/*.fits')

fig, ax = plt.subplots(figsize = (5,0.9*5*sp.sqrt(2)))

SNR = 75

z = 0

for idx in range(len(files)):
    
    with fits.open(files[idx]) as hdulist:
        flux = hdulist[0].data[0]
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        CLS = hdulist[0].header['CLASS']
        SCLS = hdulist[0].header['SUBCLASS'][0]
        try:
            U = hdulist[0].header['SNRU']
            G = hdulist[0].header['SNRG']
            R = hdulist[0].header['SNRR']
            I = hdulist[0].header['SNRI']
            Z = hdulist[0].header['SNRZ']
        except:
            U = hdulist[0].header['SN_U']
            G = hdulist[0].header['SN_G']
            R = hdulist[0].header['SN_R']
            I = hdulist[0].header['SN_I']
            Z = hdulist[0].header['SN_Z']
        #print('{}, {}, {}'.format(idx, CLS, SCLS))
    
    wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
    
    wavelength = wavelength[:-100]
    flux = flux[:-100]
    
    flux = flux/sp.amax(flux)
    '''
    if (U>=SNR or G>=SNR or R>=SNR or I>=SNR or Z>=SNR) and CLS == 'GALAXY':
         print(z)
         plt.plot(wavelength, flux)
         plt.show()
    ''' 
    z+=1
    ax.plot(wavelength, flux + idx, c = '#1f77b4')

ax.set_title('Galaxy Spectra')
ax.set_xlabel('Wavelength \ Angstroms')
ax.set_ylabel('Normalised Flux')
plt.yticks([]," ")
#ax.set_yticklabels([])
#ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig('Galaxy.pdf')
plt.show()