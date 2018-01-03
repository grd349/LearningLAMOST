#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
import pandas as pd

class Spectrum():
    ''' A class to read in a process a fits file contains a LAMOST spectrum'''
    def __init__(self, fits_sfile, fdict={'cAll':[0,9000], 'cB':[3980, 4920], 'cV':[5070,5950]}):
        self.fits_sfile = fits_sfile
        self.fdict = fdict
        keys = [n[0] for n in fdict.items()]
        keys = keys + ['FILENAME', 'DESIG']
        self.df = pd.DataFrame(columns=keys)
        
    def read_fits_file(self):
        ''' Read in and store the fits file data '''
        hdulist = fits.open(self.fits_sfile)
        self.flux = (hdulist[0].data)[0]
        self.spec_class = hdulist[0].header['CLASS']
        self.fname = hdulist[0].header['FILENAME']
        self.designation = hdulist[0].header['DESIG'][7:]
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']      
        self.wavelength = 10**(np.arange(init,init+disp*(len(self.flux)-0.9),disp))
        hdulist.close()

    def process_fits_file(self):
        ''' Kills negative flux values and deals with echelle overlap region @ ~5580 A '''
        self.flux[self.flux < 0] = np.nan
        self.flux[(self.wavelength > 5570) & (self.wavelength < 5590)] = np.nan

    def get_features(self, verbose=False):
        feats = np.zeros(len(self.fdict))
        i = 0
        for feat, lam in self.fdict.items():
            sel = np.where(np.logical_and(self.wavelength > lam[0], self.wavelength < lam[1]))
            if feat[0]=='c': 
                feats[i] = np.nanmean(self.flux[sel])
                i += 1
                if verbose:
                    print('Feature ' + feat + ' : ', np.nanmean(self.flux[sel]))
            elif feat[0]=='l':
                wavelength_range = self.wavelength[sel[0][-1]-1]-self.wavelength[sel[0][0]]
                line_area = np.trapz(self.flux[sel],self.wavelength[sel])  
                theory_area = (self.flux[sel[0][0]]+self.flux[sel[0][-1]-1]) * wavelength_range/2
                equivalent_width = (theory_area-line_area)/theory_area * wavelength_range
                feats[i] = equivalent_width
                i += 1
                if verbose:
                    print('Feature ' + feat + ' : ', equivalent_width)     
        self.df.loc[len(self.df)] = [*feats, self.fname, self.designation]
        return self.df
    
    def plot_flux(self):
        fig, ax = plt.subplots()
        ax.plot(self.wavelength, self.flux)
        plt.show()

    def __call__(self):
        self.read_fits_file()
        self.process_fits_file()
        return self.get_features()
        
if __name__ == "__main__":
    sdir = '/data2/mrs493/DR1_2/'
    files = glob.glob(sdir + '*.fits')
    fdict = {'cAll':[0,9000], 'cB':[3980, 4920], 'cV':[5070,5950], 'cR':[5890,7270], 'cI':[7310,8810],
             'lHa':[6555,6575], 'lHb':[4855,4870], 'lHg':[4320,4370]}
    i = 0
    for idx, f in enumerate(files):
        spec = Spectrum(f,fdict)
        df = spec()
        if idx == 0:
            df_main = df.copy()
        else:
            df_main = pd.concat([df_main, df])
        i += 1
        if i%100 == 0:
            print(i)
    df_main.to_csv('feature_dataframe')
    print(df_main)
