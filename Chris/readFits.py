import matplotlib.pyplot as plt
import numpy as np
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
        self.totCounts = np.sum(self.flux)
        
        width = 10
        
        self.fluxSmooth = convolve(self.flux,Box1DKernel(width))[5*width:-5*width]
        
        self.flux = self.flux[5*width:-5*width]
        self.AND = self.AND[5*width:-5*width]
        self.OR = self.OR[5*width:-5*width]
        
        #Creates a wavelength array using the central wavelength of the first pixel and the dispersion per pixel
        init = hdulist[0].header["COEFF0"]
        disp = hdulist[0].header["COEFF1"]      
        self.wavelength = 10**(np.arange(init,init+disp*(len(self.flux)-0.9),disp))
        hdulist.close()
        
        #Defines a set of wavelength for different colour bands
        letters = {"B":[3980,4920], "V":[5070,5950],"R":[5890,7270],"I":[7310,8810]}
        bandMean = {"B":0, "V":0, "R":0, "K":0}
        
        #Defines a wavelength range for characteristic absorption lines in the spectrum
        self.lines = {'Iron':[3800, 3900]}
        
        #Finds indices of upper and lower limits for colour bands and sums the counts between these limits
        for letter in letters:
            lower = np.searchsorted(self.wavelength,letters[letter][0],side="left")
            upper = np.searchsorted(self.wavelength,letters[letter][1],side="right")
            bandFlux = self.fluxSmooth[lower:upper]
            bandFlux[bandFlux<0] = np.nan
            bandMean[letter] = np.nanmean(bandFlux)
            
        #Calculates colour-colour features
        self.BV = colourFeature(bandMean["B"],bandMean["V"],self)
        self.BR = colourFeature(bandMean["B"],bandMean["R"],self)
        self.BI = colourFeature(bandMean["B"],bandMean["I"],self)
        self.VR = colourFeature(bandMean["V"],bandMean["R"],self)
        self.VI = colourFeature(bandMean["V"],bandMean["I"],self)
        self.RI = colourFeature(bandMean["R"],bandMean["I"],self)
        
        #Chooses a region of the spectrum to investigate abundance of spectral lines
        lowerFlux = np.searchsorted(self.wavelength,5000,side="left")
        upperFlux = np.searchsorted(self.wavelength,7000,side="right")
        midFlux = self.fluxSmooth[lowerFlux:upperFlux]
            
        self.spike = np.median(np.diff(midFlux[::10]))
               
        if self.spike != self.spike:
            self.VALID = False
                
    #Defines a method which plots the spectrum with the option of an inset plot showing a characteristic line
    def plotFlux(self, inset=None):    
        fig, ax1 = plt.subplots(figsize=[5,4])
        ax1.plot(self.wavelength,self.flux)
        ax1.plot(self.wavelength,self.fluxSmooth)
        #ax1.plot(self.wavelength,blackbody(9000))
        ax1.set_xlabel('Wavelength [Angstroms]')
        ax1.set_ylabel('Flux')
        ax1.set_title("Class {}, Designation {}".format(self.CLASS,self.DESIG))
        #ax1.set_title("Class {}, Temperature {}K".format(self.CLASS,self.T))
        ax1.set_yscale('log')
        
        if inset in self.lines:
            ax2 = fig.add_axes([0.25,0.25,0.25,0.25])
            ax2.plot(self.wavelength,self.flux)
            ax2.set_title(inset)
            ax2.set_xlim(self.lines[inset])
            ax2.set_yscale('log')	
            
        plt.show()
        #plt.savefig("SpectrumGap")
        
#Defines a class which holds objects made by the spectrum class and forms a dataframe using their
#header information. It then merges this dataframe with the DR1 catalog.
class Spectra:
    def __init__(self,DR1,catalog):
        
        #Reads in the catalog and removes duplicate spectra with the same designation value
        self.df = pd.read_csv(catalog, sep='|')
        self.df.drop_duplicates(subset='designation', inplace=True)
    
        self.specList = np.array([])
        
        BVList = np.array([])
        BRList = np.array([])
        BIList = np.array([])
        VRList = np.array([])
        VIList = np.array([])
        RIList = np.array([])
        self.totCountsList = np.array([])
        spikeList = np.array([])
        
        self.nameList = np.array([])
        self.desigList = np.array([])
        
        #Cycles through each spectrum, adding their header information to arrays
        for fitsName in glob.glob(DR1):
            if Spectrum(fitsName).VALID:
                self.specList = np.append(self.specList,Spectrum(fitsName))
                
                BVList = np.append(BVList,self.specList[-1].BV)
                BRList = np.append(BRList,self.specList[-1].BR)
                BIList = np.append(BIList,self.specList[-1].BI)
                VRList = np.append(VRList,self.specList[-1].VR)
                VIList = np.append(VIList,self.specList[-1].VI)
                RIList = np.append(RIList,self.specList[-1].RI)
                self.totCountsList = np.append(self.totCountsList,self.specList[-1].totCounts)
                spikeList = np.append(spikeList,self.specList[-1].spike)
                
                self.nameList = np.append(self.nameList,self.specList[-1].NAME)
                self.desigList = np.append(self.desigList,self.specList[-1].DESIG)
        
        #Creates a dataframe using the arrays
        df_spectra = pd.DataFrame(columns=['designation','BV','BR','BI','VR','VI','RI','totCounts','spike','filename'])
        for i in range(len(self.desigList)):
            df_spectra.loc[len(df_spectra)] = [self.desigList[i], BVList[i], BRList[i], BIList[i], VRList[i], VIList[i],
                           RIList[i], self.totCountsList[i], spikeList[i], self.nameList[i]]
        
        #Merges the spectra dataframe with the catalog dataframe by matching designation values then
        #writes this information to a csv file
        self.df = self.df.merge(df_spectra, on='designation', how='inner')
        self.df.to_csv('spectra_dataframe.csv')
    
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

    
    
#Constructs the Spectra variable from the DR1 data
#spec = Spectra('/data2/mrs493/DR1/*.fits','/data2/cpb405/dr1_stellar.csv')

#for i in range(10):
 #   spec.plotFlux(i)



