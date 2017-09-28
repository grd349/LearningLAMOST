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
		plt.figure()
		plt.plot(self.wavelength,self.flux)
		plt.xlabel('Wavelength \ Angstroms')
		plt.ylabel('Flux')
		plt.title(self.t_ID)
		plt.show()

spectra = []

for fitsName in glob.glob('../Data/relearninglamost/*.fits'):
	spectra.append(Spectrum(fitsName))

for i in spectra:
	i.plotFlux()
