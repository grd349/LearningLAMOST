from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

import glob

'''
O 436
B 582
A 745
F 766
G 596
K 759
M 306
'''

'''
O 476, 8773, 9818
B 96, 378, 462, 489, 492
A 17, 114, 120, 136
F 52, 158
G 25, 27, 30, 85
K 61, 65
M 256, 291, 300
'''

i = [476, 378, 17, 158, 30, 61, 256]

c = ['O', 'B', 'A', 'F', 'G', 'K', 'M'][::-1]

loc = 5891

files = [glob.glob('/data2/cpb405/Training_2/*.fits')[j] for j in i][::-1]

fig, ax = plt.subplots(figsize = (5,0.9*5*sp.sqrt(2)))

ax.axvline(6565, c = 'r', alpha = 0.1)
ax.text(6600, 7, 'Ha', color = 'r')
ax.axvline(4862, c = 'r', alpha = 0.1)
ax.text(4900, 7, 'Hb', color = 'r')
ax.axvline(4342, c = 'r', alpha = 0.1)
ax.text(4400, 7, 'Hg', color = 'r')

for idx in range(len(files)):
    
    with fits.open(files[idx]) as hdulist:
        flux = hdulist[0].data[0]
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        CLS = hdulist[0].header['CLASS']
        SCLS = hdulist[0].header['SUBCLASS'][0]
        #print('{}, {}, {}'.format(idx, CLS, SCLS))
    
    wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
    
    wavelength = wavelength[:-100]
    flux = flux[:-100]
    
    flux = sp.array(flux)

    wi = sp.searchsorted(wavelength, loc)
    
    #wi = -1
    
    flux = flux/sp.amax(flux)
        
    ax.plot(wavelength, flux + idx, label = c[idx], c = '#1f77b4')
    
    ax.annotate(c[idx], xy = (wavelength[sp.argmax(flux)]-75, idx+1.03))


ax.set_title('Stellar Spectra')
ax.set_xlabel('Wavelength \ Angstroms')
ax.set_ylabel('Normalised Flux')
plt.yticks([]," ")
#ax.set_yticklabels([])
#ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig('MK.pdf')
plt.show()