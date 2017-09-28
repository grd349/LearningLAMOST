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
        self.t_ID = hdulist[0].header['T_INFO']
        self.SNR = hdulist[0].header['SN_U']
        self.wavelength = sp.linspace(3690, 9100, len(self.flux))
        hdulist.close()

    def plotFlux(self):
        fig, ax = plt.subplots()
        ax.plot(self.wavelength,self.flux)
        ax.set_xlabel('Wavelength \ Angstroms')
        ax.set_ylabel('Flux')
        ax.set_title(self.t_ID)
        ax.set_yscale('log')
        plt.show()

spectra = []

for fitsName in glob.glob('../Data/relearninglamost/*.fits'):
    spectra.append(Spectrum(fitsName))

for i in spectra:
    i.plotFlux()
