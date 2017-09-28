import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.io import fits

class Spectrum:
    def __init__(self,path):
	hdulist = fits.open(path)
	self.flux = (hdulist[0].data)[0]
	self.date = hdulist[0].header["DATE"]
	self.t_ID = hdulist[0].header["T_INFO"]
	self.SNR = hdulist[0].header["SN_U"]
	self.wavelength = np.linspace(3690,9000,len(self.flux))
	hdulist.close()

    def plotFlux(self):
	fig, ax = plt.subplots()
        ax.plot(self.wavelength,self.flux)
        ax.set_xlabel('Wavelength [Angstroms]')
        ax.set_ylabel('Flux')
        ax.set_title(self.t_ID)
        ax.set_yscale('log')
        plt.show()

spectra = []

for fitsName in glob.glob('../Data/relearninglamost/*.fits'):
    spectra.append(Spectrum(fitsName))

for spectrumNumber in spectra:
    spectrumNumber.plotFlux()


	




