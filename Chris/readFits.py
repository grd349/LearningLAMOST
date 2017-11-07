import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.io import fits
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
        self.CLASS = hdulist[0].header["CLASS"]
        self.NAME = hdulist[0].header["FILENAME"]
        self.DESIG = hdulist[0].header["DESIG"][7:]
        self.totCounts = np.sum(self.flux)
        
        #Creates a wavelength array using the central wavelength of the first pixel and the dispersion per pixel
        init = hdulist[0].header["COEFF0"]
        disp = hdulist[0].header["COEFF1"]      
        self.wavelength = 10**(np.arange(init,init+disp*(len(self.flux)-0.9),disp))
        hdulist.close()
        
        #Uses the equation for a blackbody to calculate an "ideal" model using the wavelength values of the actual spectrum
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        self.T = 15000
        E = 1e-4*(8*np.pi*h*c)/((self.wavelength*1e-10)**5*(np.exp(h*c/((self.wavelength*1e-10)*k*self.T))-1))
        
        #Normalises the model
        self.normalise = self.totCounts/np.sum(E)
        self.bbFlux = self.normalise*E
        
        #Defines a set of wavelength for different colour bands
        letters = {"B":[3980,4920], "V":[5070,5950],"R":[5890,7270], "K":[19950,23850]}
        bandMean = {"B":0, "V":0, "R":0, "K":0}
        
        #Defines a wavelength range for characteristic absorption lines in the spectrum
        self.lines = {'Iron':[3800, 3900]}
        
        #Finds indices of upper and lower limits for colour bands and sums the counts between these limits
        for letter in letters:
            lower = np.searchsorted(self.wavelength,letters[letter][0],side="left")
            upper = np.searchsorted(self.wavelength,letters[letter][1],side="right")
            bandFlux = self.flux[lower:upper]
            bandFlux[bandFlux<0] = np.nan
            bandMean[letter] = np.nanmean(bandFlux)
            
        #Calculates a B-V feature
        self.BminusV = (-2.5 * np.log10(bandMean["B"]))-(-2.5 * np.log10(bandMean["V"]))
        self.BminusR = (-2.5 * np.log10(bandMean["B"]))-(-2.5 * np.log10(bandMean["R"]))
        self.VminusR = (-2.5 * np.log10(bandMean["V"]))-(-2.5 * np.log10(bandMean["R"]))
        
        if self.BminusR == np.nan or self.BminusR == (-1)*np.inf or self.VminusR == np.nan or self.VminusR == (-1)*np.inf:
            print(self.NAME)
            self.VALID = False
                
    #Defines a method which plots the spectrum with the option of an inset plot showing a characteristic line
    def plotFlux(self, inset=None):    
        fig, ax1 = plt.subplots(figsize=[5,4])
        ax1.plot(self.wavelength,self.flux)
        #ax1.plot(self.wavelength,self.bbFlux)
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
            
        #plt.show()
        plt.savefig("SpectrumGap")
        
#Defines a class which holds objects made by the spectrum class and forms a dataframe using their
#header information. It then merges this dataframe with the DR1 catalog.
class Spectra:
    def __init__(self,DR1,catalog):
        
        #Reads in the catalog and removes duplicate spectra with the same designation value
        self.df = pd.read_csv(catalog, sep='|')
        self.df.drop_duplicates(subset='designation', inplace=True)
    
        self.specList = np.array([])
        self.BminusVList = np.array([])
        self.BminusRList = np.array([])
        self.VminusRList = np.array([])
        self.nameList = np.array([])
        self.totCountsList = np.array([])
        self.desigList = np.array([])
        
        #Cycles through each spectrum, adding their header information to arrays
        for fitsName in glob.glob(DR1):
            if Spectrum(fitsName).VALID:
                self.specList = np.append(self.specList,Spectrum(fitsName))
                self.BminusVList = np.append(self.BminusVList,self.specList[-1].BminusV)
                self.BminusRList = np.append(self.BminusRList,self.specList[-1].BminusR)
                self.VminusRList = np.append(self.VminusRList,self.specList[-1].VminusR)
                self.nameList = np.append(self.nameList,self.specList[-1].NAME)
                self.totCountsList = np.append(self.totCountsList,self.specList[-1].totCounts)
                self.desigList = np.append(self.desigList,self.specList[-1].DESIG)
        
        #Creates a dataframe using the arrays
        df_spectra = pd.DataFrame(columns=['designation', 'BminusV', 'BminusR', 'VminusR', 'totCounts', 'filename'])
        for i in range(len(self.desigList)):
            df_spectra.loc[len(df_spectra)] = [self.desigList[i], self.BminusVList[i], self.BminusRList[i], self.VminusRList[i], self.totCountsList[i], self.nameList[i]]
        
        #Merges the spectra dataframe with the catalog dataframe by matching designation values then
        #writes this information to a csv file
        self.df = self.df.merge(df_spectra, on='designation', how='inner')
        self.df.to_csv('spectra_dataframe.csv')
    
    #Calls the plot method in the Spectrum class
    def plotFlux(self,specNumber):
        self.specList[specNumber].plotFlux()
        
#Constructs the Spectra variable from the DR1 data
#spec = Spectra('/data2/mrs493/DR1/*.fits','/data2/cpb405/dr1_stellar.csv')



