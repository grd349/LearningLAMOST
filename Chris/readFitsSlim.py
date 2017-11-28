import numpy as np
import glob
from astropy.io import fits

#Defines a class which stores a single spectrum, extracts information about this spectrum from its header file and
#calculates features using colour filters.
class Spectrum:
    def __init__(self,DR1):
        #Load in spectrum
        hdulist = fits.open(DR1)
        
        #Extract information from header file
        self.flux = (hdulist[0].data)[0]
        self.DESIG = hdulist[0].header["DESIG"][7:]  
    
        #Creates a wavelength array using the central wavelength of the first pixel and the dispersion per pixel
        init = hdulist[0].header["COEFF0"]
        disp = hdulist[0].header["COEFF1"]      
        self.wavelength = 10**(np.arange(init,init+disp*(len(self.flux)-0.9),disp))
        hdulist.close()
        
class Spectra:
    def __init__(self,DR1):             
        self.specList = np.array([])
        self.desig = np.array([]) 
        
        for fitsName in glob.glob(DR1):
            self.specList = np.append(self.specList,Spectrum(fitsName))
            self.desig = np.append(self.desig,self.specList[-1].DESIG)


