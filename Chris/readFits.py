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
        self.CLASS = hdulist[0].header["CLASS"]
        self.SN_U = hdulist[0].header["SN_U"]
        self.SN_G = hdulist[0].header["SN_G"]
        self.SN_R = hdulist[0].header["SN_R"]
        self.SN_I = hdulist[0].header["SN_I"]
        self.SN_Z = hdulist[0].header["SN_Z"]
        self.DESIG = hdulist[0].header["DESIG"][7:]

        self.totCounts = np.sum(self.flux)
        
        init = hdulist[0].header["COEFF0"]
        disp = hdulist[0].header["COEFF1"]
        
        self.wavelength = 10**(np.arange(init,init+disp*(len(self.flux)-0.9),disp))
        hdulist.close()
        
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        self.T = 3000
        E = 1e-4*(8*np.pi*h*c)/((self.wavelength*1e-10)**5*(np.exp(h*c/((self.wavelength*1e-10)*k*self.T))-1))
        
        self.fudge = self.totCounts/np.sum(E)
        self.bbFlux = self.fudge*E
        
        self.letters = {"B":[3980,4920], "V":[5070,5950],"R":[5890,7270]}
        self.bandCounts = {"B":0, "V":0, "R":0}
        
        self.lines = {'Iron':[3800, 3900]}
        
        for letter in self.letters:
            lower = np.searchsorted(self.wavelength,self.letters[letter][0],side="left")
            upper = np.searchsorted(self.wavelength,self.letters[letter][1],side="right")       
            self.bandCounts[letter] = np.sum(self.flux[lower:upper])
            
        self.BminusV = np.log10(self.bandCounts["B"])-np.log10(self.bandCounts["V"])

    def plotFlux(self, inset=None):    
        fig, ax1 = plt.subplots()
        ax1.plot(self.wavelength,self.flux)
        ax1.plot(self.wavelength,self.bbFlux)
        ax1.set_xlabel('Wavelength [Angstroms]')
        ax1.set_ylabel('Flux')
        ax1.set_title("Class {}, ID {}, Temperature {}K".format(self.CLASS,self.t_ID,self.T))
        ax1.set_yscale('log')

        if inset in self.lines:
            ax2 = fig.add_axes([0.6,0.55,0.25,0.25])
            ax2.plot(self.wavelength,self.flux)
            ax2.set_title(inset)
            ax2.set_xlim(self.lines[inset])
            ax2.set_yscale('log')	
            
        plt.show()
        #plt.savefig("Spectrum3")
        

class Spectra:
    def __init__(self,path):
        self.specList = np.array([])
        self.colourList = np.array([])
        self.totCountsList = np.array([])
        self.desigList = np.array([])
        self.fluxList = []
        self.wavelengthList = []
        
        for fitsName in glob.glob(path):
            self.specList = np.append(self.specList,Spectrum(fitsName))
            self.colourList = np.append(self.colourList,self.specList[-1].BminusV)
            self.totCountsList = np.append(self.totCountsList,self.specList[-1].totCounts)
            self.desigList = np.append(self.desigList,self.specList[-1].DESIG)
            self.fluxList.append(self.specList[-1].flux)
            self.wavelengthList.append(self.specList[-1].wavelength)
            
        self.fluxList = np.array(self.fluxList)
        self.wavelengthList = np.array(self.wavelengthList)
        
    def plotFlux(self,specNumber):
        self.specList[specNumber].plotFlux()
"""            
spec = Spectra('../Data/DR1/*.fits')

print(spec.desigList)
#plt.plot(spec.wavelengthList[0],spec.fluxList[0])

spec.plotFlux(0)
    
fig, ax1 = plt.subplots()
ax1.scatter(spec.colourList,spec.totCountsList)
ax1.set_xlabel('B-V Feature')
ax1.set_ylabel('Total Counts')
ax1.set_title("Scatter Plot of Total Counts against B-V Feature")
#ax1.set_yscale('log')
plt.show()
#plt.savefig("FeaturePlot2")
"""	




