#!/usr/bin/env python3

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
import seaborn #for plotting, to be used later

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

sfile = '/data2/mrs493/dr1_stellar.csv'

catalog = pd.read_csv(sfile, sep='|')

catalog.drop_duplicates(subset = 'designation', inplace = True)

dr1 = pd.DataFrame(columns = ['designation', 'flux', 'wavelength', 'totalCounts', 'B', 'V', 'R', 'colour'])

letters = {'B':[3980, 4920], 'V':[5070, 5950], 'R':[5890, 7270]}

for fitsName in glob.glob('../Data/DR1/*.fits'):
    hdulist = fits.open(fitsName)
    
    init = hdulist[0].header['COEFF0']
    disp = hdulist[0].header['COEFF1']
    flux = hdulist[0].data[0]
    
    wavelength = 10**sp.arange(init, init+disp*(len(flux)-0.9), disp)
    totalCounts = sp.sum(flux)
    
    bands = {'B':0, 'V':0, 'R':0}
            
    for letter in letters:
        lower = sp.searchsorted(wavelength, letters[letter][0], side = 'left')
        upper = sp.searchsorted(wavelength, letters[letter][1], side = 'right')
        bands[letter] = sp.sum(flux[lower:upper])
        
    colour = sp.log10(bands['B']/bands['V'])
    
    dr1.loc[len(dr1)] = [hdulist[0].header['DESIG'][7:], flux, wavelength, totalCounts, bands['B'], bands['V'], bands['R'], colour]
    hdulist.close()

df = catalog.merge(dr1, on='designation', how='inner')


"""
change above to a function, or write to a csv to be used in future code
"""


fig, ax = plt.subplots()
ax.scatter(df.colour, df.teff)
ax.set_title('Star colour vs. temperature')
ax.set_xlabel('Star colour / log(B/V)')
ax.set_ylabel('Temperature / K')

colour = sp.reshape(df.colour, (-1, 1))
temperature = []

for i in range(len(df.teff)):
    temperature.append(int(df.teff[i]))
        #the temperatures need to be integers

temperature= sp.array(temperature)

clf = RandomForestClassifier()
kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)

"""

errors = []
var = []

for train_index, test_index in kf:
    X_train, X_test = colour[train_index], colour[test_index]
    y_train, y_test = temperature[train_index], temperature[test_index]
    clf = clf.fit(X_test,y_test)
    test_pred = clf.predict(X_test)
    error = abs(test_pred - y_test)
    #errors.append(sp.mean(error))
    errors.append(error)
    
print sp.mean(error)

#need to add cross validation
"""


clf.fit(colour, temperature)
pred = clf.predict(colour)

error = pred - temperature

fig, ax = plt.subplots()
ax.scatter(temperature, pred)
ax.set_xlabel('Actual temperature \ K')
ax.set_ylabel('Predicted temperature \ K')
ax.set_title('Actual vs. Predicted temperature')

fig, ax = plt.subplots()
ax.hist(error)
ax.set_title('Error of Prediction')



"""
if __name__ == "__main__":
    sfile = '/data2/mrs493/dr1_stellar.csv'

    df = pd.read_csv(sfile, sep='|')
    print(df.columns)
    
    print(df.loc[0:10].obsid)

    print(df.loc[df.obsid == 101005].teff)

    # Panadas make a dataframe ...

    df_spectra = pd.DataFrame(columns=['obsid', 'feature'])
    ids = [(101001, 3), (101005, 4), (101008, 10)]
    for i in ids:
        df_spectra.loc[len(df_spectra)] = [i[0], i[1]]

    print(df_spectra)
    df = df.merge(df_spectra, on='obsid', how='inner')

    print(df)

    df.to_csv('my_data.csv')
    
    fig, ax = plt.subplots()
    ax.hist(df.teff)
    plt.show()
    fig, ax = plt.subplots()
    ax.hist(df['feh'])
    plt.show()
"""