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

fBands = {'cB':[3980,4920], 'cV':[5070,5950],'cR':[5890,7270],'cI':[7310,8810],
          'lHa':[6555, 6575], 'lHb':[4855, 4870], 'lHg':[4320,4370],
          'lHd':[4093,4113], 'lHe':[3960,3980], 'lNa':[5885,5905],
          'lMg':[5167,5187], 'lK':[3925,3945], 'lG':[4240,4260],
          'lFe':[5160,5180]}

keys = ['designation', 'CLASS', 'filename', 'total', 'd1', 'd2', 'd3'] + [feat[0] for feat in fBands.items()]

errors = pd.DataFrame(columns = ['file'])

dr1 = pd.DataFrame(columns=keys)

for idx, fitsName in enumerate(files):
    
    if idx%100 == 0:
        print(idx)
    
    hdulist = fits.open(fitsName)
    try:
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        flux = hdulist[0].data[0]
    
        wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
    
        stitchLower = sp.searchsorted(wavelength,5570,side='left')
        stitchUpper = sp.searchsorted(wavelength,5590,side='right')
        flux[stitchLower:stitchUpper] = sp.nan
    	
        '''
        starts
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
    
        '''
        end
        '''
    
        values = sp.zeros(len(fBands))
        i = 0
        
        for feat in fBands:
            wBound = fBands[feat]
            wLower = sp.searchsorted(wavelength, wBound[0], side = 'left')
            wUpper = sp.searchsorted(wavelength, wBound[1], side = 'right')
    
            if feat[0]=='l':
                ends = [flux[wLower], flux[wUpper - 1]]
                wRange = wavelength[wUpper-1] - wavelength[wLower]
                
                actualA = sp.trapz(flux[wLower:wUpper], wavelength[wLower:wUpper])
                
                fW = sp.concatenate((wavelength[wLower-20:wLower], wavelength[wUpper-1:wUpper+19]))
                fF = sp.concatenate((flux[wLower-20:wLower], flux[wUpper-1:wUpper+19]))
                
                nans = sp.logical_not(sp.isnan(fF))
                fW = fW[nans]
                fF = fF[nans]
                '''
                S = sp.sum(1/fF)
                Sx = sp.sum(fW/fF)
                Sy = len(fF)
                Sxx = sp.sum(fW**2/fF)
                Sxy = sp.sum(fW)
                
                delta = S*Sxx - Sx**2
                a = (Sxx*Sy - Sx*Sxy)/delta
                b = (S*Sxy-Sx*Sy)/delta
                theoA = (b*( wavelength[wUpper-1] + wavelength[wLower] ) + 2*a)*wRange/2.###
                '''
                sLin = sp.polyfit(fW, fF, 1)
                
                theoA = (sLin[0]*( wavelength[wUpper-1] + wavelength[wLower] ) + 2*sLin[1])*wRange/2.
        
                values[i] = wRange*(1-(actualA/theoA))
                                    
            elif feat[0]=='c':
                bandFlux = flux[wLower:wUpper]
                values[i] = -2.5*sp.log10(sp.nanmean(bandFlux))
    
            if values[i] != values[i] or abs(values[i]) == sp.inf:
                values[i] = 0 #need to think of better fix
            
            i += 1
            
        df = pd.DataFrame(columns=keys)
        df.loc[0] = [hdulist[0].header['DESIG'][7:], hdulist[0].header['CLASS'], hdulist[0].header['FILENAME'], total, diff1, diff2, diff3, *values]
        dr1 = pd.concat([dr1, df])

    except:
        print('error reading file ', files[idx])
        er = pd.DataFrame(columns = ['file'])
        er.loc[0] = [files[idx]]
        errors = pd.concat([errors, er])

    hdulist.close()
    gc.collect()

dr1.to_csv('spectra5.csv', index = False)

errors.to_csv('errors.csv', index = False)
