#!/usr/bin/env python3

#see how useful smoothed differences are

import pandas as pd
import scipy as sp
#import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

import gc
import glob

files = glob.glob('/data2/mrs493/DR1_2/*.fits')

fBands = {'cB':[3980,4920], 'cV':[5070,5950],'cR':[5890,7270],'cI':[7310,8810], 'lHa':[6555, 6575], 'lHb':[4855, 4870], 'lHg':[4320,4370]}

keys = ['designation', 'CLASS', 'filename', 'total', 'd1', 'd2', 'd3'] + [feat[0] for feat in fBands.items()]

errors = pd.DataFrame(columns = ['file'])

dr1 = pd.DataFrame(columns=keys)

for idx, fitsName in enumerate(files):
    
    if idx%100 == 0:
        print idx
    
    hdulist = fits.open(fitsName)
    try:
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        flux = hdulist[0].data[0]
    
        wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
        
        stitchLower = sp.searchsorted(wavelength,5570,side='left')
        stitchUpper = sp.searchsorted(wavelength,5590,side='right')
        flux[stitchLower:stitchUpper] = sp.nan
        
        ##
        '''re-add'''
            
        
        '''
        start
        '''
        
        wid = 10
        width = 100
        buff = 1
        
        smth = convolve(flux,Box1DKernel(wid))[buff*width:-buff*width]
        smoothFlux = convolve(flux,Box1DKernel(width))[buff*width:-buff*width]
        
        flux[flux<0] = sp.nan
        smth[smth<0] = sp.nan
        smoothFlux[smoothFlux<0] = sp.nan
        
        total = sp.nanmean(flux)
        
        diff1 = sp.nanmean(abs(flux[buff*width:-buff*width] - smoothFlux))/total
        diff2 = sp.nanmean(abs(flux[buff*width:-buff*width] - smth))/total
        diff3 = sp.nanmean(abs(smth - smoothFlux))/total
        
        #
        
        ##flux = flux[5*width:-5*width]
        ##wavelength = wavelength[5*width:-5*width]
        
        #spike = sp.median(sp.diff(flux[::10]))
    
        #testing = sp.diff(flux[::10])
        #testing2 = (testing==testing and abs(testing)>10)
        #counts = [abs(testing)]   
            #to do: look into 'spikeness' 
        '''
        end
        '''
        
        values = sp.zeros(len(fBands))
        i = 0
        
        for feat in fBands:
            wRange = fBands[feat]
            wLower = sp.searchsorted(wavelength, wRange[0], side = 'left')
            wUpper = sp.searchsorted(wavelength, wRange[1], side = 'right')
    
            if feat[0]=='l':
                ends = [flux[wLower], flux[wUpper - 1]]
                wRange = wavelength[wUpper-1] - wavelength[wLower]
    
                actualA = sp.trapz(flux[wLower:wUpper], wavelength[wLower:wUpper])
                theoA = (ends[0] + ends[1])*wRange/2.
    
                values[i] = wRange*(1-(actualA/theoA))
       
            elif feat[0]=='c':
                bandFlux = flux[wLower:wUpper]
                values[i] = -2.5*sp.log10(sp.nanmean(bandFlux))

            if values[i] != values[i] or abs(values[i]) == sp.inf:
                values[i] = 0 #need to think of better fix
                
            i += 1
    
        df = pd.DataFrame(columns=keys)
        df.loc[0] = [hdulist[0].header['DESIG'][7:], hdulist[0].header['CLASS'], hdulist[0].header['FILENAME'], total, diff1, diff2, diff3, values[0], values[1], values[2], values[3], values[4], values[5], values[6]] #upgrade to using python 3 and use values* instead of individual indexing
        dr1 = pd.concat([dr1, df])

    except:
        df = pd.DataFrame(columns = ['file'])
        df.loc[0] = [hdulist[0].header['FILENAME']]
        errors = pd.concat([errors, df])
    
    hdulist.close()
    gc.collect()

dr1.to_csv('spectra.csv', index = False)

errors.to_csv('errors.csv', index = False)
