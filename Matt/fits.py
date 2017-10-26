#!/usr/bin/env python3

from astropy.io import fits
import scipy as sp
import matplotlib.pyplot as plt
import glob

class Spectrum:
    #a class to read and store information from the .fits files of DR1 spectra
    def __init__(self, path):
        #takes the file path of the .fits file as an argument
        hdulist = fits.open(path)
            #open the .fits file to allow for data access
        self.flux = hdulist[0].data[0]  #flux counts of the spectra
        self.date = hdulist[0].header['DATE']   #date the observation was made
        self.SPID = hdulist[0].header['SPID']   #spectral ID (all = 6 that we've seen)
        self.SNR_U = hdulist[0].header['SN_U']  #signal to noise ration (SNR) in U band
        self.SNR_G = hdulist[0].header['SN_G']  #SNR in G band
        self.SNR_R = hdulist[0].header['SN_R']  #SNR in R band
        self.SNR_I = hdulist[0].header['SN_I']  #SNR in I band
        self.SNR_Z = hdulist[0].header['SN_Z']  #SNR in Z band
        self.CLASS = hdulist[0].header['CLASS'] #object LAMOST classification
        
        self.desig = hdulist[0].header['DESIG'][7:] #Designation of the object
        
        self.totCounts = sp.sum(self.flux)  #Sum the total counts to give a feature
        	
        init = hdulist[0].header['COEFF0']
            #coeff0 is the centre point of the first point in log10 space
        disp = hdulist[0].header['COEFF1']
            #coeff1 is the seperation between points in log10 space
        
        self.wavelength = 10**sp.arange(init, init+disp*(len(self.flux)-0.9), disp)
            #use coeff0 and coeff1 to calculate the wavelength of each pixel in angstroms
        	
        hdulist.close()
            #close the .fits file
        self.lines = {'Iron':[3800, 3900]}
            #elements, and the window in which their emmision lines are seen 
        self.letters = {'B':[3980, 4920], 'V':[5070, 5950], 'R':[5890, 7270]}
            #colour bands and their corresponding wavelength windows
        self.bands = {'B':0, 'V':0, 'R':0}
            #colour bands and the (to be calculated) total counts in that band 
        
        for letter in self.letters:
            lower = sp.searchsorted(self.wavelength, self.letters[letter][0], side = 'left')
                #find the index of the lower boundary of the band
            upper = sp.searchsorted(self.wavelength, self.letters[letter][1], side = 'right')
                #find the index of the upper boundary of the band
            self.bands[letter] = -2.5*sp.log10(sp.sum(self.flux[lower:upper]))
                #total the counts between these index, and convert to a colour
        
        self.colour = self.bands['B'] - self.bands['V']
            #store the difference between the B and V bands as a feature
        
    def plotFlux(self, ax = None, T = None, element = None, titles = False):
        #method to plot the spectra and scaled blackbody curve, and also zoom in on element lines
        if not ax: fig, ax = plt.subplots()
        
        ax.plot(self.wavelength,self.flux)
        
        if T:
            h = 6.63e-34
            c = 3e8
            k = 1.38e-23
            
            E = (8*sp.pi*h*c)/((self.wavelength*1e-10)**5*(sp.exp(h*c/((self.wavelength*1e-10)*k*T))-1))
            #Calculate an ideal black body curve for a temperature T

            fudge = self.totCounts/sp.sum(E)
                #normalise blackbody curve by scaling by the total counts ratio of the curve to the spectra        
            self.bbFlux = fudge*E
                #the normalised blackbody curve
            
            ax.plot(self.wavelength,self.bbFlux)
                #plot the flux and blackbody curve against wavelength
        if titles:
            ax.set_xlabel('Wavelength \ Angstroms')
            ax.set_ylabel('Flux')
            ax.set_yscale('log')
            ax.set_title('{} - {}'.format(self.SPID, self.CLASS))

        if element in self.lines:
		#plot inset plot for selected element
            ax1 = fig.add_axes([0.6,0.55,0.25,0.25])
            ax1.plot(self.wavelength,self.flux)
            ax1.set_title(element)
            ax1.set_xlim(self.lines[element])
            ax1.set_xticks(self.lines[element])
            ax1.set_yscale('log')
