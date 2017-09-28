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
	self.SN_U = hdulist[0].header["SN_U"]
	self.SN_G = hdulist[0].header["SN_G"]
	self.SN_R = hdulist[0].header["SN_R"]
	self.SN_I = hdulist[0].header["SN_I"]
	self.SN_Z = hdulist[0].header["SN_Z"]
	
	init = hdulist[0].header["COEFF0"]
	disp = hdulist[0].header["COEFF1"]

	self.wavelength = 10**(np.arange(init,init+disp*len(self.flux),disp))[0:len(self.flux)]
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


	




