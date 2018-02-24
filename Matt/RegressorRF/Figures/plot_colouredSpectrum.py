from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

import glob

#matplotlib.rcParams.update({'font.size': 22})

#files = glob.glob('/data2/mrs493/DR1_3/*.fits')
#files = [glob.glob('/data2/mrs493/DR1_3/*.fits')[9]]
files = [glob.glob('/data2/mrs493/DR1_3/*.fits')[17]]
#files = [glob.glob('/data2/mrs493/DR1_3/*.fits')[12]]
#files = [glob.glob('/data2/mrs493/DR1_3/*.fits')[2]]

cBands = {'cB':[3980,4920], 'c#820BBB':[5070,5950],'cR':[5890,7270],'c#2E0854':[7310,8810]}
"""        
lBands = {'lHa':[6555, 6575], 'lHb':[4855, 4870], 'lHg':[4320,4370],
          'lHd':[4093,4113], 'lHe':[3960,3980], 'lNa':[5885,5905],
          'lMg':[5167,5187], 'lK':[3925,3945], 'lG':[4240,4260]}
"""
#lBands = {'lHa':[6555, 6575], 'lHb':[4855, 4870], 'lHg':[4320,4370]}

lBands = {'lHa':[6555, 6575]}

for idx in range(len(files)):

    print(idx)

    fig, ax = plt.subplots(1,2)
    
    with fits.open(files[idx]) as hdulist:
        flux = hdulist[0].data[0]
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        
    wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
    """
    ax.plot(wavelength, flux, color='k')
    
    ax.set_ylim(plt.ylim())
    ax.set_xlim(plt.xlim())
    
    ax.set_xlabel('Wavelength / Angstroms')
    ax.set_ylabel('Flux')
    
    for col in cBands:
        ax.fill_betweenx(plt.ylim(), cBands[col][0], cBands[col][1], facecolor = col[1:], alpha=0.5)
    for lin in lBands:
        wBound = lBands[lin]
        wLower = sp.searchsorted(wavelength, wBound[0], side = 'left')
        wUpper = sp.searchsorted(wavelength, wBound[1], side = 'right')
        
        ends = [flux[wLower], flux[wUpper - 1]]
        wRange = wavelength[wUpper-1] - wavelength[wLower]
        
        actualA = sp.trapz(flux[wLower:wUpper], wavelength[wLower:wUpper])
        
        fW = sp.concatenate((wavelength[wLower-20:wLower], wavelength[wUpper-1:wUpper+19]))
        fF = sp.concatenate((flux[wLower-20:wLower], flux[wUpper-1:wUpper+19]))
        
        nans = sp.logical_not(sp.isnan(fF))
        fW = fW[nans]
        fF = fF[nans]
        
        sLin = sp.polyfit(fW, fF, 1)

        fig, ax = plt.subplots()
        
        ax.plot(wavelength[wLower:wUpper-1], flux[wLower:wUpper-1], color = 'k')
        ax.plot(wavelength[wLower-20:wLower+1], flux[wLower-20:wLower+1], color = 'g')
        ax.plot(wavelength[wUpper-2:wUpper+19], flux[wUpper-2:wUpper+19], color = 'g')
        ax.plot(wavelength[wLower-20:wUpper+19], sLin[0]*wavelength[wLower-20:wUpper+19] + sLin[1], color = 'r')
        
        ax.set_xlabel('Wavelength / Angstroms')
        ax.set_ylabel('Flux')
        
        #ax.fill_betweenx(plt.ylim(), cBands[col][0], cBands[col][1], facecolor = col[1:], alpha=0.5)

    wid = 10
    width = 100
    buff = 1
    
    smth = convolve(flux,Box1DKernel(wid))[buff*width:-buff*width]
    smoothFlux = convolve(flux,Box1DKernel(width))[buff*width:-buff*width]
    
    flx = flux[buff*width:-buff*width]
    
    wav = wavelength[buff*width:-buff*width]
    
    fig, ax = plt.subplots()
    
    ax.plot(wav, flx)
    ax.plot(wav, smth)
    ax.plot(wav, smoothFlux)
    
    ax.set_xlabel('Wavelength / Angstroms')
    ax.set_ylabel('Flux')
    """
    
    ax[0].plot(wavelength, flux)
    ax[0].set_xlabel('Wavelength / Angstroms')
    ax[0].set_ylabel('Flux')
    ax[0].set_title('Unprocessed LAMOST Spectrum')
        
    x = ax[0].get_xlim()
    y = ax[0].get_ylim()
    
    ax[0].set_ylim(y)


    wBound = [6555, 6575]
    #ax[0].fill_betweenx(y, wBound[0], wBound[1], facecolor = 'g', alpha=0.5)
    wLower = sp.searchsorted(wavelength, wBound[0], side = 'left')
    wUpper = sp.searchsorted(wavelength, wBound[1], side = 'right')
    
    ends = [flux[wLower], flux[wUpper - 1]]
    wRange = wavelength[wUpper-1] - wavelength[wLower]
    
    actualA = sp.trapz(flux[wLower:wUpper], wavelength[wLower:wUpper])
    
    fW = sp.concatenate((wavelength[wLower-20:wLower], wavelength[wUpper-1:wUpper+19]))
    fF = sp.concatenate((flux[wLower-20:wLower], flux[wUpper-1:wUpper+19]))
    
    nans = sp.logical_not(sp.isnan(fF))
    fW = fW[nans]
    fF = fF[nans]
    
    sLin = sp.polyfit(fW, fF, 1)
    
    ax[1].plot(wavelength[wLower:wUpper-1], flux[wLower:wUpper-1], color = 'k')
    ax[1].plot(wavelength[wLower-20:wLower+1], flux[wLower-20:wLower+1], color = 'g')
    ax[1].plot(wavelength[wUpper-2:wUpper+19], flux[wUpper-2:wUpper+19], color = 'g')
    ax[1].plot(wavelength[wLower-20:wUpper+19], sLin[0]*wavelength[wLower-20:wUpper+19] + sLin[1], color = 'r')
    
    ax[1].set_xlabel('Wavelength / Angstroms')
    ax[1].set_ylabel('Flux')
    ax[1].set_title('H-alpha')
    
    plt.tight_layout()
    plt.show()