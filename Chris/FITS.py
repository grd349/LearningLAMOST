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
		plt.plot(self.wavelength,self.flux)
		plt.xlabel("Wavelength [Angstroms]")
		plt.ylabel("Flux")
		plt.title("ID {}, SNR {}, Date {}".format(self.t_ID, self.SNR, self.date))
		plt.show()

spectra = []

for fitsName in glob.glob('../Data/relearninglamost/*.fits'):

	spectra.append(Spectrum(fitsName))

for spectrumNumber in spectra:
	spectrumNumber.plotFlux()


	




