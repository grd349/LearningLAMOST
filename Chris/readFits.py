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
            
        self.colour = np.log(self.bandCounts["B"])-np.log(self.bandCounts["V"])


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
        

spectra = []

for fitsName in glob.glob('../Data/DR1/*.fits'):
    spectra.append(Spectrum(fitsName))

#for spectrumNumber in spectra:
 #   spectrumNumber.plotFlux()

spectra[27].plotFlux()

colour = []
counts = []

for spectrum in spectra:
    colour.append(spectrum.colour)
    counts.append(spectrum.totCounts)
    
fig, ax1 = plt.subplots()
ax1.scatter(colour,counts)
ax1.set_xlabel('B-V Feature')
ax1.set_ylabel('Total Counts')
ax1.set_title("Scatter Plot of Total Counts against B-V Feature")
#ax1.set_yscale('log')
plt.show()
#plt.savefig("FeaturePlot2")
	




