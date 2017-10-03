import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.io import fits

class Spectrum:
    def __init__(self,path):
        hdulist = fits.open(path)
        self.flux = (hdulist[0].data)[0]
        self.date = hdulist[0].header["DATE"]
        self.t_ID = hdulist[0].header["SPID"]
        self.CLASS = hdulist[0].header["CLASS"]
        self.SN_U = hdulist[0].header["SN_U"]
        self.SN_G = hdulist[0].header["SN_G"]
        self.SN_R = hdulist[0].header["SN_R"]
        self.SN_I = hdulist[0].header["SN_I"]
        self.SN_Z = hdulist[0].header["SN_Z"]
        
        init = hdulist[0].header["COEFF0"]
        disp = hdulist[0].header["COEFF1"]
        
        self.wavelength = 10**(np.arange(init,init+disp*(len(self.flux)-0.9),disp))
        hdulist.close()

    def plotFlux(self, inset=None):
        
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        T = 7000
        E = 1e-4*(8*np.pi*h*c)/((self.wavelength*1e-10)**5*(np.exp(h*c/((self.wavelength*1e-10)*k*T))-1))
        
        fig, ax1 = plt.subplots()
        ax1.plot(self.wavelength,self.flux)
        ax1.plot(self.wavelength,E)
        ax1.set_xlabel('Wavelength [Angstroms]')
        ax1.set_ylabel('Flux')
        ax1.set_title("Class {}, ID {}".format(self.CLASS,self.t_ID))
        ax1.set_yscale('log')

        features = {'Iron':[3800, 3900]}

        if inset in features:
            ax2 = fig.add_axes([0.6,0.55,0.25,0.25])
            ax2.plot(self.wavelength,self.flux)
            ax2.set_title(inset)
            ax2.set_xlim(features[inset])
            ax2.set_yscale('log')	
            
        plt.show()

spectra = []

for fitsName in glob.glob('../Data/DR1/*.fits'):
    spectra.append(Spectrum(fitsName))

for spectrumNumber in spectra:
    spectrumNumber.plotFlux('Iron')


	




