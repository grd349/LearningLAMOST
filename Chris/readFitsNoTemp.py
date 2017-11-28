import matplotlib.pyplot as plt
import numpy as np
import gc
import glob
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel
import pandas as pd

#Defines a class which stores a single spectrum, extracts information about this spectrum from its header file and
#calculates features using colour filters.
class Spectrum:
    def __init__(self,DR1):
        #Load in spectrum
        hdulist = fits.open(DR1)
        
        self.VALID = True
        
        #Extract information from header file
        self.flux = (hdulist[0].data)[0]
        self.AND = (hdulist[0].data)[3]
        self.OR = (hdulist[0].data)[4]
        self.CLASS = hdulist[0].header["CLASS"]
        self.NAME = hdulist[0].header["FILENAME"]
        self.DESIG = hdulist[0].header["DESIG"][7:]
        
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
        self.AND = self.AND[5*width:-5*width]
        self.OR = self.OR[5*width:-5*width]
        self.wavelength = self.wavelength[5*width:-5*width]
        
        artifLower = np.searchsorted(self.wavelength,5570,side="left")
        artifUpper = np.searchsorted(self.wavelength,5590,side="right")
        
        self.flux[artifLower:artifUpper] = np.nan
        
        self.totCounts = np.nansum(self.flux)
        
        #Defines a set of wavelength for different colour bands
        letters = {"B":[3980,4920], "V":[5070,5950],"R":[5890,7270],"I":[7310,8810]}
        bandMean = {}
        
        #Defines a wavelength range for characteristic absorption lines in the spectrum
        self.lines = {'Iron':[3800, 3900]}
        
        #Finds indices of upper and lower limits for colour bands and sums the counts between these limits
        for letter in letters:
            lower = np.searchsorted(self.wavelength,letters[letter][0],side="left")
            upper = np.searchsorted(self.wavelength,letters[letter][1],side="right")
            
            #Calculates a mean number of counts ignoring NaNs for each colour band
            bandFlux = self.flux[lower:upper]
            bandMean[letter] = np.nanmean(bandFlux)
            
        #Produces colour minus colour features
        self.BV = colourFeature(bandMean["B"],bandMean["V"],self)
        self.BR = colourFeature(bandMean["B"],bandMean["R"],self)
        self.BI = colourFeature(bandMean["B"],bandMean["I"],self)
        self.VR = colourFeature(bandMean["V"],bandMean["R"],self)
        self.VI = colourFeature(bandMean["V"],bandMean["I"],self)
        self.RI = colourFeature(bandMean["R"],bandMean["I"],self)
        
        self.HalphaEW = specLine([6555,6575],self)
        self.HbetaEW = specLine([4855,4870],self)
        self.HgammaEW = specLine([4320,4370],self)
                          
    #Defines a method which plots the spectrum with the option of an inset plot showing a characteristic line
    def plotFlux(self, inset=None):    
        fig, ax1 = plt.subplots(figsize=[5,4])
        ax1.plot(self.wavelength,self.flux)
        ax1.axvline(4471)
        ax1.plot(self.wavelength,self.fluxSmooth)
        ax1.set_xlabel('Wavelength [Angstroms]')
        ax1.set_ylabel('Flux')
        ax1.set_title("Class {}, Designation {}".format(self.CLASS,self.DESIG))
        ax1.set_yscale('log')
        ax1.set_xlim([4450,4490])
        
        if inset in self.lines:
            ax2 = fig.add_axes([0.25,0.25,0.25,0.25])
            ax2.plot(self.wavelength,self.flux)
            ax2.set_title(inset)
            ax2.set_xlim(self.lines[inset])
            ax2.set_yscalartifLower = np.searchsorted(self.wavelength,5570,side="left")
            
        plt.show()
        
#Defines a class which holds objects made by the spectrum class and forms a dataframe using their
#header information. It then merges this dataframe with the DR1 catalog.
class Spectra:
    def __init__(self,DR1):
               
        self.specList,BV,BR,BI,VR,VI,RI,totCounts,HalphaEW,HbetaEW,HgammaEW,name,desig,CLASS=(np.array([]) for i in range(14))
        
        #Cycles through each spectrum, adding their header information to arrays
        for fitsName in glob.glob(DR1):
            if Spectrum(fitsName).VALID:
                self.specList = np.append(self.specList,Spectrum(fitsName))
                
                BV = np.append(BV,self.specList[-1].BV)
                BR = np.append(BR,self.specList[-1].BR)
                BI = np.append(BI,self.specList[-1].BI)
                VR = np.append(VR,self.specList[-1].VR)
                VI = np.append(VI,self.specList[-1].VI)
                RI = np.append(RI,self.specList[-1].RI)
                totCounts = np.append(totCounts,self.specList[-1].totCounts)
                HalphaEW = np.append(HalphaEW,self.specList[-1].HalphaEW)
                HbetaEW = np.append(HbetaEW,self.specList[-1].HbetaEW)
                HgammaEW = np.append(HgammaEW,self.specList[-1].HgammaEW)
                
                name = np.append(name,self.specList[-1].NAME)
                desig = np.append(desig,self.specList[-1].DESIG)
                CLASS = np.append(CLASS,self.specList[-1].CLASS)
                
                gc.collect()
        
        #Creates a dataframe using the arrays
        df_spectra = pd.DataFrame(columns=['designation','BV','BR','BI','VR','VI','RI','totCounts','HalphaEW','HbetaEW','HgammaEW','filename','class'])
        for i in range(len(desig)):
            df_spectra.loc[len(df_spectra)] = [desig[i], BV[i], BR[i], BI[i], VR[i], VI[i],
                           RI[i], totCounts[i], HalphaEW[i], HbetaEW[i], HgammaEW[i], name[i], CLASS[i]]
        
        df_spectra.to_csv('spectra_dataframe_no_temp.csv')
    
    #Calls the plot method in the Spectrum class
    def plotFlux(self,specNumber):
        self.specList[specNumber].plotFlux()

def blackbody(T,wavelength,Spectrum):
    #Calculates the flux and wavelength for the blackbody
    h = 6.63e-34
    c = 3e8
    k = 1.38e-23
    E = 1e-4*(8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))

    #Normalises the model
    normalise = Spectrum.totCounts/np.sum(E)
    return normalise*E

def colourFeature(col1,col2,Spectrum):
    #Calculate a subtraction feature from two colour indexes
    feature = (-2.5 * np.log10(col1))-(-2.5 * np.log10(col2))
    
    if feature != feature or feature == (-1)*np.inf or feature == np.inf:
            Spectrum.VALID = False
            
    return feature

def specLine(wavelengths,Spectrum):
    wavLower = np.searchsorted(Spectrum.wavelength,wavelengths[0],side="left")
    wavUpper = np.searchsorted(Spectrum.wavelength,wavelengths[1],side="right")
    
    actualArea = np.trapz(Spectrum.flux[wavLower:wavUpper],Spectrum.wavelength[wavLower:wavUpper])
    theoryArea = (Spectrum.flux[wavLower]+Spectrum.flux[wavUpper-1])*(Spectrum.wavelength[wavUpper-1]-Spectrum.wavelength[wavLower])/2.
    
    equivWidth = (theoryArea-actualArea)/theoryArea * (Spectrum.wavelength[wavUpper-1]-Spectrum.wavelength[wavLower])
    
    if equivWidth != equivWidth:
            Spectrum.VALID = False
            
    return equivWidth
    
#Constructs the Spectra variable from the DR1 data
spec = Spectra('/data2/mrs493/DR1_3/*.fits')
"""
for i in range(100):
    spec.plotFlux(i)
    #print(spec.specList[i].equivWidth)
"""


