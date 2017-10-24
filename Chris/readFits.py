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
        
        #Extract information from header file
        self.flux = (hdulist[0].data)[0]
        self.CLASS = hdulist[0].header["CLASS"]
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
        self.letters = {"B":[3980,4920], "V":[5070,5950],"R":[5890,7270]}
        self.bandCounts = {"B":0, "V":0, "R":0}
        
        #Defines a wavelength range for characteristic absorption lines in the spectrum
        self.lines = {'Iron':[3800, 3900]}
        
        #Finds indices of upper and lower limits for colour bands and sums the counts between these limits
        for letter in self.letters:
            lower = np.searchsorted(self.wavelength,self.letters[letter][0],side="left")
            upper = np.searchsorted(self.wavelength,self.letters[letter][1],side="right")       
            self.bandCounts[letter] = np.sum(self.flux[lower:upper])
            
        #Calculates a B-V feature
        self.BminusV = (-2.5 * np.log10(self.bandCounts["B"]))-(-2.5 * np.log10(self.bandCounts["V"]))

    #Defines a method which plots the spectrum with the option of an inset plot showing a characteristic line
    def plotFlux(self, inset=None):    
        fig, ax1 = plt.subplots(figsize=[5,4])
        ax1.plot(self.wavelength,self.flux)
        ax1.plot(self.wavelength,self.bbFlux)
        ax1.set_xlabel('Wavelength [Angstroms]')
        ax1.set_ylabel('Flux')
        #ax1.set_title("Class {}, Designation {}".format(self.CLASS,self.DESIG))
        ax1.set_title("Class {}, Temperature {}K".format(self.CLASS,self.T))
        ax1.set_yscale('log')

        if inset in self.lines:
            ax2 = fig.add_axes([0.25,0.25,0.25,0.25])
            ax2.plot(self.wavelength,self.flux)
            ax2.set_title(inset)
            ax2.set_xlim(self.lines[inset])
            ax2.set_yscale('log')	
            
        plt.show()
        #plt.savefig("Spectrum5")
        
#Defines a class which holds objects made by the spectrum class and forms a dataframe using their
#header information. It then merges this dataframe with the DR1 catalog.
class Spectra:
    def __init__(self,DR1,catalog):
        
        #Reads in the catalog and removes duplicate spectra with the same designation value
        self.df = pd.read_csv(catalog, sep='|')
        self.df.drop_duplicates(subset='designation', inplace=True)
    
        self.specList = np.array([])
        self.colourList = np.array([])
        self.totCountsList = np.array([])
        self.desigList = np.array([])
        self.fluxList = []
        self.wavelengthList = []
        
        #Cycles through each spectrum, adding their header information to arrays
        for fitsName in glob.glob(DR1):
            self.specList = np.append(self.specList,Spectrum(fitsName))
            self.colourList = np.append(self.colourList,self.specList[-1].BminusV)
            self.totCountsList = np.append(self.totCountsList,self.specList[-1].totCounts)
            self.desigList = np.append(self.desigList,self.specList[-1].DESIG)
            self.fluxList.append(self.specList[-1].flux)
            self.wavelengthList.append(self.specList[-1].wavelength)

        self.fluxList = np.array(self.fluxList)
        self.wavelengthList = np.array(self.wavelengthList)
        
        #Creates a dataframe using the arrays
        df_spectra = pd.DataFrame(columns=['designation', 'feature', 'wavelength', 'flux'])
        for i in range(len(self.desigList)):
            df_spectra.loc[len(df_spectra)] = [self.desigList[i], self.colourList[i], self.wavelengthList[i], self.fluxList[i]]
        
        #Merges the spectra dataframe with the catalog dataframe by matching designation values then
        #writes this information to a csv file
        self.df = self.df.merge(df_spectra, on='designation', how='inner')
        self.df.to_csv('spectra_dataframe.csv')
    
    #Calls the plot method in the Spectrum class
    def plotFlux(self,specNumber):
        self.specList[specNumber].plotFlux()
        
#Constructs the Spectra variable from the DR1 data
spec = Spectra('/data2/cpb405/DR1/*.fits','/data2/cpb405/dr1_stellar.csv')

#print spec.df["flux"]

#spec.plotFlux(70)
"""
fig, ax1 = plt.subplots()
ax1.scatter(spec.colourList,spec.totCountsList)
ax1.set_xlabel('B-V Feature')
ax1.set_ylabel('Total Counts')
ax1.set_title("Scatter Plot of Total Counts against B-V Feature")
#plt.savefig("TotCountsColPlot")
plt.show()

fig, ax2 = plt.subplots()
ax2.scatter([spec.df.feature], spec.df.teff)
ax2.set_xlabel('B-V Feature')
ax2.set_ylabel('Effective Temperature / K')
ax2.set_title('Plot of Spectral Temperature Against B-V Feature')
#plt.savefig("Tempcolplot")
plt.show()
"""




