#!/usr/bin/env python3

from astropy.io import fits
import scipy as sp
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel

class Spectrum:
    #a class to read and store information from the .fits files of DR1 spectra
    def __init__(self, path):
        #takes the file path of the .fits file as an argument
        
        width = 10 #not decided on value yet
        
        hdulist = fits.open(path)
            #open the .fits file to allow for data access
        self.flux = hdulist[0].data[0]  #flux counts of the spectra
        self.date = hdulist[0].header['DATE']   #date the observation was made

        self.CLASS = hdulist[0].header['CLASS'] #object LAMOST classification
        
        self.smoothFlux = convolve(self.flux,Box1DKernel(width))[5*width:-5*width]
        
        self.desig = hdulist[0].header['DESIG'][7:] #Designation of the object
        
        self.totCounts = sp.sum(self.flux)  #Sum the total counts to give a feature
        	
        init = hdulist[0].header['COEFF0']
            #coeff0 is the centre point of the first point in log10 space
        disp = hdulist[0].header['COEFF1']
            #coeff1 is the seperation between points in log10 space
        
        self.wavelength = 10**sp.arange(init, init+disp*(len(self.flux)-0.9), disp)[5*width:-5*width]
            #use coeff0 and coeff1 to calculate the wavelength of each pixel in angstroms
        
        self.flux = self.flux[5*width: -5*width]
        
        hdulist.close()
            #close the .fits file
        self.lines = {'Iron':[3800, 3900]}
            #elements, and the window in which their emmision lines are seen 
        self.letters = {"B":[3980,4920], "V":[5070,5950],"R":[5890,7270],"I":[7310,8810]}
            #colour bands and their corresponding wavelength windows
        self.bands = {"B":0, "V":0, "R":0, "K":0}
            #colour bands and the (to be calculated) total counts in that band 
        
        for letter in self.letters:
            lower = sp.searchsorted(self.wavelength, self.letters[letter][0], side = 'left')
                #find the index of the lower boundary of the band
            upper = sp.searchsorted(self.wavelength, self.letters[letter][1], side = 'right')
                #find the index of the upper boundary of the band
            bandFlux = self.smoothFlux[lower:upper]
            bandFlux[bandFlux<0] = sp.nan
            self.bands[letter] = -2.5*sp.log10(sp.nanmean(bandFlux))
        
        self.BV = self.bands['B'] - self.bands['V']
        self.BR = self.bands['B'] - self.bands['R']
        self.BI = self.bands['B'] - self.bands['I']
        self.VR = self.bands['V'] - self.bands['R']
        self.VI = self.bands['V'] - self.bands['I']
        self.RI = self.bands['R'] - self.bands['I']
        
    def plotFlux(self, ax = None, Tpred = None, Teff = None, element = None, colour = '#1f77b4', label = None, log = True):
        #method to plot the spectra and scaled blackbody curve, and also zoom in on element lines
        if not ax: fig, ax = plt.subplots()
        
        ax.plot(self.wavelength,self.flux, color = colour, label = label)
        
        if Tpred:
            h = 6.63e-34
            c = 3e8
            k = 1.38e-23
            
            E = (8*sp.pi*h*c)/((self.wavelength*1e-10)**5*(sp.exp(h*c/((self.wavelength*1e-10)*k*Tpred))-1))
            #Calculate an ideal black body curve for a temperature T

            fudge = self.totCounts/sp.sum(E)
                #normalise blackbody curve by scaling by the total counts ratio of the curve to the spectra        
            self.bbFlux = fudge*E
                #the normalised blackbody curve
            
            ax.plot(self.wavelength,self.bbFlux, ls = '--', label = 'Predicted', color = 'r')
                #plot the flux and blackbody curve against wavelength
        
        if Teff:
            h = 6.63e-34
            c = 3e8
            k = 1.38e-23
            
            E = (8*sp.pi*h*c)/((self.wavelength*1e-10)**5*(sp.exp(h*c/((self.wavelength*1e-10)*k*Teff))-1))
            #Calculate an ideal black body curve for a temperature T

            fudge = self.totCounts/sp.sum(E)
                #normalise blackbody curve by scaling by the total counts ratio of the curve to the spectra        
            self.bbFlux = fudge*E
                #the normalised blackbody curve
            
            ax.plot(self.wavelength,self.bbFlux, ls = ':', label = 'Effective', color = 'g')
                #plot the flux and blackbody curve against wavelength

        if log: ax.set_yscale('log')

        if element in self.lines:
		#plot inset plot for selected element
            ax1 = fig.add_axes([0.6,0.55,0.25,0.25])
            ax1.plot(self.wavelength,self.flux)
            ax1.set_title(element)
            ax1.set_xlim(self.lines[element])
            ax1.set_xticks(self.lines[element])
            ax1.set_yscale('log')
