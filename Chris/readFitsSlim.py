import numpy as np
import glob
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

#Defines a class which stores a single spectrum, extracts information about this spectrum from its header file and
#calculates features using colour filters.
class Spectrum:
    def __init__(self,DR1):
        #Load in spectrum
        hdulist = fits.open(DR1)
        
        #Extract information from header file
        self.flux = (hdulist[0].data)[0]
        
        width = 10
        
        #NaNs every flux that is less than zero (these are artificial)
        self.flux[self.flux < 0] = np.nan
        
        #Smooths the spectrum to remove noise by convolving it with a running boxcar function
        self.fluxSmooth = convolve(self.flux,Box1DKernel(width))[5*width:-5*width]     
    
        #Creates a wavelength array using the central wavelength of the first pixel and the dispersion per pixel
        init = hdulist[0].header["COEFF0"]
        disp = hdulist[0].header["COEFF1"]      
        self.wavelength = 10**(np.arange(init,init+disp*(len(self.flux)-0.9),disp))
        hdulist.close()
        
        #Trims the ends of the arrays to remove edge effects from the boxcar function
        self.flux = self.flux[5*width:-5*width]
        self.wavelength = self.wavelength[5*width:-5*width]
        
        artifLower = np.searchsorted(self.wavelength,5570,side="left")
        artifUpper = np.searchsorted(self.wavelength,5590,side="right")
        
        self.flux[artifLower:artifUpper] = np.nan
        
class Spectra:
    def __init__(self,DR1):             
        self.specList=np.array([])
        
        for fitsName in glob.glob(DR1):
            self.specList = np.append(self.specList,Spectrum(fitsName))


