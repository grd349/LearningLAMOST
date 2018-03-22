import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
import pandas as pd
from read_fits import Spectrum
from matplotlib.lines import Line2D

'''  MK: 361, 588, 143, 635, 561, 54, 1457     '''

"""
sdir = '/data2/cpb405/QSO/'
files = glob.glob(sdir + '*.fits')

flux = []
wavelength = []
scls = []
redshift = []

for idx, file in enumerate(files):
    with fits.open(file) as hdulist:
        f = hdulist[0].data[0]
        f = f/np.sum(f)
        z = hdulist[0].header['Z']
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        w = 10**(np.arange(init,init+disp*(len(f)-0.9),disp))

    flux.append(f[:-100])
    wavelength.append(w[:-100])
    redshift.append(z)

"QSOS"
fig, ax = plt.subplots(figsize=(6,10))

indexes = [9,2,5,0,4,6,7]
for i in range(len(indexes)):
    ax.plot(wavelength[indexes[i]],i*0.0011+flux[indexes[i]], color = 'k')
    ax.text(wavelength[indexes[i]][-2600],(i*0.0011+0.0003)+flux[indexes[i]][-2600],'z = {:.2f}'.format(redshift[indexes[i]]),horizontalalignment='center', color = 'blueviolet')

ax.set_xlabel("Wavelength [Angstroms]")
ax.set_ylabel("Shifted Normalised Flux")

plt.tick_params(
    axis='y',
    left='off', 
    labelleft='off') 

plt.show()
"""





direc = 'TempCSVs1'

features_df = pd.concat([pd.read_csv(f, sep=',') for f in glob.glob(direc + '/*.csv')])       
catalog_file = "/data2/cpb405/dr1_stellar.csv"
catalog = pd.read_csv(catalog_file, sep='|')
catalog.drop_duplicates(subset = 'designation', inplace = True)
df = catalog.merge(features_df, on='designation', how='inner')
df = df.sort_values('cAll')[:1000]

temp = df['teff'].values
desig = df['designation'].values
logg = df['logg'].values
Mg = df['lMg'].values
Ha = df['lHa'].values
Na = df['lNa'].values
"""
for i in range(len(temp)):
    if temp[i] < 5000 and logg[i]>4:
        print(logg[i])
        print(i)
        fig, ax = plt.subplots()
        row = df.loc[df['designation']==desig[i]].index[0]
        spec = Spectrum('/data2/mrs493/DR1_2/' + df.get_value(row,'FILENAME'))
    
        f = spec.flux
        ax.plot(spec.wavelength, f, color='k')
        ax.set_xlim([5100,5254])
        ax.axvline(x=5167, color = 'blueviolet')
        ax.axvline(x=5187, color = 'blueviolet')  
        plt.show()


l = []
mg = []
ha = []
na = []
col = []
for i in range(len(temp)):
    l.append(logg[i])
    mg.append(Mg[i])
    ha.append(Ha[i])
    na.append(Na[i])
    if temp[i] > 7500:
        c = 'b'
    elif temp[i] > 6000:
        c = 'r'
    elif temp[i] > 5000:
        c = 'g'
    elif temp[i] > 3500:
        c = 'blueviolet'
    col.append(c)

custom_lines = [Line2D([0], [0], marker='o', color='b', label='A',
                          markerfacecolor='b', markersize=5),
                Line2D([0], [0], marker='o', color='r', label='F',
                          markerfacecolor='r', markersize=5),
                Line2D([0], [0], marker='o', color='g', label='G',
                          markerfacecolor='g', markersize=5),
                Line2D([0], [0], marker='o', color='blueviolet', label='K',
                          markerfacecolor='blueviolet', markersize=5)]
                
fig, ax = plt.subplots(3,1,figsize=(6,10),sharey=True)
ax[0].scatter(mg,l,s=5,color=col)
ax[0].set_xlabel('Mg Equivalent Width [Angstroms]')
ax[0].set_ylabel('Surface Gravity')
ax[0].legend(['A','F','G','K'],['b','r','g','blueviolet'])
ax[0].legend(handles=custom_lines)
      
ax[1].scatter(ha,l,s=5,color=col)
ax[1].set_xlabel('H-alpha Equivalent Width [Angstroms]')
ax[1].set_ylabel('Surface Gravity')

ax[2].scatter(na,l,s=5,color=col)
ax[2].set_xlabel('Na Width Equivalent [Angstroms]')
ax[2].set_ylabel('Surface Gravity')

plt.tight_layout()
plt.show()

"""

fig, ax = plt.subplots(figsize=(6,10))

indexes = [258,141,174,47,51,226,68,61,59]
for i in range(len(indexes)):
    row = df.loc[df['designation']==desig[indexes[i]]].index[0]
    spec = Spectrum('/data2/mrs493/DR1_2/' + df.get_value(row,'FILENAME'))  
    
    f = spec.flux
    while f[np.searchsorted(spec.wavelength,5177)] > 0:
        f = f - 10
    ax.plot(spec.wavelength, i*1000+f, color='k')
    ax.text(5120,(i*1000+400),'Logg = {0:.2f}'.format(logg[indexes[i]]),horizontalalignment='center', color = 'blueviolet')
 
ax.axvline(x=5167, color = 'blueviolet')
ax.axvline(x=5187, color = 'blueviolet')   
ax.set_xlim([5100,5254])
ax.set_xlabel("Wavelength [Angstroms]")
ax.set_ylabel("Shifted Flux")

plt.tick_params(
    axis='y',
    left='off', 
    labelleft='off')
plt.show()  



"""
sdir = '/data2/cpb405/Spec_Types/'
files = glob.glob(sdir + '*.fits')

flux = []
wavelength = []
for idx, file in enumerate(files):
    with fits.open(file) as hdulist:
        f = hdulist[0].data[0]
        f = f
        init = hdulist[0].header['COEFF0']
        disp = hdulist[0].header['COEFF1']
        w = 10**(np.arange(init,init+disp*(len(f)-0.9),disp))

    flux.append(f[:-100])
    wavelength.append(w[:-100])

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, figsize=(6,10),sharex=True)
ax1.plot(wavelength[2], flux[2], color = 'k')
ax2.plot(wavelength[0], flux[0], color = 'k')
ax3.plot(wavelength[4], flux[4], color = 'k')
ax4.plot(wavelength[6], flux[6], color = 'k')
ax5.plot(wavelength[1], flux[1], color = 'k')
ax6.plot(wavelength[5], flux[5], color = 'k')
ax7.plot(wavelength[3], flux[3], color = 'k')

ax1.annotate('Absorption Galaxy', xy=(0.6, 0.2), xycoords='axes fraction', color='blueviolet')
ax2.annotate('Emission Galaxy', xy=(0.1, 0.8), xycoords='axes fraction', color='blueviolet')
ax3.annotate('Unknown', xy=(0.6, 0.8), xycoords='axes fraction', color='blueviolet')
ax4.annotate('Carbon Star', xy=(0.1, 0.8), xycoords='axes fraction', color='blueviolet')
ax5.annotate('Double Star', xy=(0.6, 0.2), xycoords='axes fraction', color='blueviolet')
ax6.annotate('Emission Line', xy=(0.1, 0.8), xycoords='axes fraction', color='blueviolet')
ax7.annotate('White Dwarf', xy=(0.6, 0.8), xycoords='axes fraction', color='blueviolet')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.text(0.5, 0.06, 'Wavelength [Angstroms]', ha='center')
f.text(0.02, 0.5, 'Flux', va='center', rotation='vertical')
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.show()
"""
