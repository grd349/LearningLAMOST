#!/usr/bin/env python3

import gc

import pandas as pd
import scipy as sp
import glob
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

import matplotlib.pyplot as plt

width = 10

sfile = '/data2/cpb405/dr1_stellar.csv'

catalog = pd.read_csv(sfile, sep='|')

catalog.drop_duplicates(subset = 'designation', inplace = True)

dr1 = pd.DataFrame(columns = ['designation', 'totalCounts', 'B', 'V', 'R', 'I', 'Ha', 'Hb', 'Hg', 'CLASS', 'filename'])

letters = {"B":[3980,4920], "V":[5070,5950],"R":[5890,7270],"I":[7310,8810]}

lines = {'Ha':[6555, 6575], 'Hb':[4855, 4870], 'Hg':[4320,4370]}

for fitsName in glob.glob('/data2/cpb405/DR1/*.fits'):
    
    hdulist = fits.open(fitsName)
    
    valid = True
            
    init = hdulist[0].header['COEFF0']
    disp = hdulist[0].header['COEFF1']
    flux = hdulist[0].data[0]

    wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
    
    stitchLower = sp.searchsorted(wavelength,5570,side="left")
    stitchUpper = sp.searchsorted(wavelength,5590,side="right")
    
    flux[stitchLower:stitchUpper] = sp.nan
        
    flux[flux<0] = sp.nan
        
    smoothFlux = convolve(flux,Box1DKernel(width))[5*width:-5*width] 
    flux = flux[5*width:-5*width]
    wavelength = wavelength[5*width:-5*width]
    
    totalCounts = sp.nansum(flux)
    #spike = sp.median(sp.diff(flux[::10]))
    
    '''
    start
    '''
    
    #testing = sp.diff(flux[::10])
    #testing2 = (testing==testing and abs(testing)>10)
#    counts = [abs(testing)]   
        #to do: look into 'spikeness' 
    
    
    '''
    end
    '''
    
    eqWid = {}
    
    for line in lines:
        wRange = lines[line]
    
        wLower = sp.searchsorted(wavelength, wRange[0], side = 'left')
        wUpper = sp.searchsorted(wavelength, wRange[1], side = 'right')
        
        ends = [flux[wLower], flux[wUpper - 1]]
        wRange = wavelength[wUpper-1] - wavelength[wLower]
    
        actualA = sp.trapz(flux[wLower:wUpper], wavelength[wLower:wUpper])
        theoA = (ends[0] + ends[1])*wRange/2.
        
        eqWid[line] = wRange*(1-(actualA/theoA))
                    
        if not eqWid[line] == eqWid[line]: valid = False
             
    bands = {}
    
    for letter in letters:
        lower = sp.searchsorted(wavelength, letters[letter][0], side = 'left')
        upper = sp.searchsorted(wavelength, letters[letter][1], side = 'right')
        bandFlux = smoothFlux[lower:upper]
        bands[letter] = -2.5*sp.log10(sp.nanmean(bandFlux))
        if bands[letter] == sp.inf or bands[letter] == -sp.inf:
            bands[letter] = sp.nan
            valid = False
                        
    if valid:
        dr1.loc[len(dr1)] = [hdulist[0].header['DESIG'][7:], totalCounts, bands['B'], bands['V'], bands['R'], bands['I'], eqWid['Ha'], eqWid['Hb'], eqWid['Hg'], hdulist[0].header['CLASS'], hdulist[0].header['FILENAME']]

    hdulist.close()
    
    gc.collect()

dr1.to_csv('/data2/mrs493/classes.csv')