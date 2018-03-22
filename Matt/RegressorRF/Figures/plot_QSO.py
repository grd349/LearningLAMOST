from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

import glob

i = [9952, 9579, 3853, 5746, 321, 581, 499][::-1]

files = [glob.glob('/data2/cpb405/Training/*.fits')[j] for j in i]
#files = glob.glob('/data2/cpb405/Training/*.fits')

fig, ax = plt.subplots(figsize = (5,0.9*5*sp.sqrt(2)))

"""
ax.axvline(6565, c = 'r', alpha = 0.1)
ax.annotate('Ha', xy = (7200,8.4), xytext = (6465, 9), color = 'r', arrowprops=dict(arrowstyle="->", color='red'))
ax.axvline(4862, c = 'g', alpha = 0.1)
#ax.text(4762, 8.8, 'Hb', color = 'r')
ax.annotate('Hb', xy = (5350,8.4), xytext = (4762, 8.8), color = 'g', arrowprops=dict(arrowstyle="->", color='green'))
ax.axvline(4342, c = 'purple', alpha = 0.1)
ax.annotate('Hg', xy = (4795, 8.4),  xytext = (4242, 9), color = 'purple', arrowprops=dict(arrowstyle="->", color='purple'))
ax.axvline(4960, c = 'orange', alpha = 0.1)
ax.axvline(5008, c = 'orange', alpha = 0.1)
ax.annotate('OIII', xy = (5510,8.3), xytext = (4858, 9), color = 'orange', arrowprops=dict(arrowstyle="->", color='orange'))
#ax.annotate('OIII', xy = (5640,8.4), xytext = (4858, 9), color = 'r', arrowprops=dict(arrowstyle="->", color='red'))
#ax.text(4858, 9, 'OIII', color = 'r')
"""
z = 0

'''
crem de la crem
spec-55892-GAC_082N27_M1_sp13-249.fits
spec-57005-M31020N36M1_sp14-206.fits
spec-55892-F9205_sp03-222.fits
'''
files = ['/data2/cpb405/Training/spec-56206-EG232011N230528B01_sp07-196.fits','/data2/cpb405/Training/spec-55913-M31_014139N300249_F1_sp03-053.fits']
'''
poss:
spec-55878-B87802_1_sp02-162.fits
spec-57043-EG030739N012421M01_sp15-224.fits
spec-55911-B91107_sp04-107.fits

'''

'''
to check:
spec-56403-HD163229N165051B01_sp03-242.fits

'''

for idx in range(len(files)):
    
    with fits.open(files[idx]) as hdulist:
        flux = hdulist[0].data[0]
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        CLS = hdulist[0].header['CLASS']
        SCLS = hdulist[0].header['SUBCLASS'][0]
        met = hdulist[0].header['Z']
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
    if I>=SNR and CLS == 'QSO':
         print(z, CLS)
         plt.plot(wavelength, flux)
         plt.show()
    '''    
    z+=1
    ax.plot(wavelength, flux + 1.3*idx, c = '#1f77b4')
    #ax.annotate('z ={}'.format(met), xy = (7500, 1.3*idx-0.1))

#ax.set_title('QSO Spectra')
ax.set_xlabel('Wavelength \ Angstroms')
ax.set_ylabel('Normalised Flux')
plt.yticks([]," ")
#ax.set_yticklabels([])
#ax.get_yaxis().set_visible(False)
plt.tight_layout()
#plt.savefig('QSO.pdf')
plt.show()
