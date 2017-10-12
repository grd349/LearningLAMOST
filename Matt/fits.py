#!/usr/bin/env python3

from astropy.io import fits
import scipy as sp
import matplotlib.pyplot as plt
import glob

class Spectrum:
    def __init__(self, path):
        hdulist = fits.open(path)
        
        self.flux = hdulist[0].data[0]
        self.date = hdulist[0].header['DATE']
        self.SPID = hdulist[0].header['SPID']
        self.SNR_U = hdulist[0].header['SN_U']
        self.SNR_G = hdulist[0].header['SN_G']
        self.SNR_R = hdulist[0].header['SN_R']
        self.SNR_I = hdulist[0].header['SN_I']
        self.SNR_Z = hdulist[0].header['SN_Z']
        self.CLASS = hdulist[0].header['CLASS']
        
        self.desig = hdulist[0].header['DESIG'][7:]
        
        self.totCounts = sp.sum(self.flux)
        	
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        
        self.wavelength = 10**sp.arange(init, init+disp*(len(self.flux)-0.9), disp)
        '../Data/DR1/*.fits'
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        T = 7000
        E = (8*sp.pi*h*c)/((self.wavelength*1e-10)**5*(sp.exp(h*c/((self.wavelength*1e-10)*k*T))-1))
        
        fudge = self.totCounts/sp.sum(E)
        
        self.bbFlux = fudge*E
        	
        hdulist.close()
        
        self.lines = {'Iron':[3800, 3900]}
        
        self.letters = {'B':[3980, 4920], 'V':[5070, 5950], 'R':[5890, 7270]}
        self.bands = {'B':0, 'V':0, 'R':0}
        
        for letter in self.letters:
            lower = sp.searchsorted(self.wavelength, self.letters[letter][0], side = 'left')
            upper = sp.searchsorted(self.wavelength, self.letters[letter][1], side = 'right')
            self.bands[letter] = sp.sum(self.flux[lower:upper])
        
        self.colour = sp.log(self.bands['B']) - sp.log(self.bands['V'])
            
        
    def plotFlux(self, element = None):

        
        fig, ax = plt.subplots()
        ax.plot(self.wavelength,self.flux)
        ax.plot(self.wavelength,self.bbFlux)
        
        ax.set_xlabel('Wavelength \ Angstroms')
        ax.set_ylabel('Flux')
        ax.set_yscale('log')
        ax.set_title('{} - {}'.format(self.SPID, self.CLASS))
        
        if element in self.lines:
            ax1 = fig.add_axes([0.6,0.55,0.25,0.25])
            ax1.plot(self.wavelength,self.flux)
            ax1.set_title(element)
            ax1.set_xlim(self.lines[element])
            ax1.set_xticks(self.lines[element])
            ax1.set_yscale('log')
            
        plt.show()
        
class Spectra:
    def __init__(self, path):
        self.spectra = sp.array([])
        self.flux = []
        self.colour = sp.array([])
        self.wavelength = []
        self.SPID = sp.array([])
        self.CLASS = sp.array([])
        
        for fitsName in glob.glob(path):
            self.spectra = sp.append(self.spectra, Spectrum(fitsName))
            self.flux.append(self.spectra[-1].flux)
            self.colour = sp.append(self.colour, self.spectra[-1].colour)
            self.wavelength.append(self.spectra[-1].wavelength)
            self.SPID = sp.append(self.SPID, self.spectra[-1].SPID)
            self.CLASS = sp.append(self.CLASS, self.spectra[-1].CLASS)
            
        self.flux = sp.array(self.flux)
        self.wavelength = sp.array(self.wavelength)
    
    def plotFlux(self, index, element = None):
        self.spectra[index].plotFlux(element)


#spectra = Spectra('../Data/DR1/*.fits')

#print spectra.colour

"""
spectra = []

for fitsName in glob.glob('../Data/DR1/*.fits'):
    spectra.append(Spectrum(fitsName))

colour = []
counts = []

for i in spectra:
    colour.append(i.colour)
    counts.append(i.totCounts)

fig, ax = plt.subplots()
ax.scatter(colour,counts)

ax.set_xlabel('log (B/V)')
ax.set_ylabel('total Counts')
#ax.set_yscale('log')
ax.set_title('Spectra Features')
plt.savefig('featurePlot')
"""
