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
        self.SNR = hdulist[0].header['SN_U']
        self.CLASS = hdulist[0].header['CLASS']
        	
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        
        self.wavelength = 10**sp.arange(init, init+disp*(len(self.flux)-0.9), disp)
        	
        hdulist.close()
    
    def plotFlux(self, element = None):
        fig, ax = plt.subplots()
        ax.plot(self.wavelength,self.flux)
        ax.set_xlabel('Wavelength \ Angstroms')
        ax.set_ylabel('Flux')
        ax.set_yscale('log')
        ax.set_title('{} - {}'.format(self.SPID, self.CLASS))
        
        lines = {'Iron':[3800, 3900]}
            #decide on a 'range' & add more elements
            #add actual line positon and plot?
        
        if element in lines:
            ax1 = fig.add_axes([0.6,0.55,0.25,0.25])
            ax1.plot(self.wavelength,self.flux)
            ax1.set_title(element)
            ax1.set_xlim(lines[element])
            ax1.set_ylim(ymin=0)
            ax1.set_xticks(lines[element])
            
        plt.show()

spectra = []

for fitsName in glob.glob('../Data/relearninglamost/*.fits'):
    spectra.append(Spectrum(fitsName))

for i in spectra:
    i.plotFlux('Iron')
