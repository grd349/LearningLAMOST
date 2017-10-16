#!/usr/bin/env python3

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
import seaborn as sns

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

plt.savefig('cvt')

#plot colour vs. temp

colour = sp.reshape(df.colour, (-1, 1))
temperature = []

for i in range(len(df.teff)):
    temperature.append(int(df.teff[i]))
        #the temperatures need to be integers

temperature= sp.array(temperature)

clf = RandomForestClassifier()
kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)

accuracy = []

for j in range(1,21):

	accuracy_sum = []
	kf = cross_validation.KFold(n = len(colour), n_folds = 5, shuffle = True)

	for train_index, test_index in kf:
	
		X_train, X_test = colour[train_index], colour[test_index]
		y_train, y_test = temperature[train_index], temperature[test_index]
		clf = clf.fit(X_train, y_train)
		test_pred = clf.predict(X_test)
		accuracy = sp.concatenate((abs(test_pred - y_test)/(y_test*1.0), accuracy))
		
mean_accuracy = sp.mean(accuracy)
std_accuracy = sp.std(accuracy)

clf.fit(colour, temperature)
pred = clf.predict(colour)

error = pred - temperature

fig, ax = plt.subplots()
ax.scatter(temperature, pred)
ax.set_xlabel('Actual temperature \ K')
ax.set_ylabel('Predicted temperature \ K')
ax.set_title('Actual vs. Predicted temperature')

ax.text(10000, 4000, '{} +/- {}'.format(mean_accuracy, std_accuracy), size = 15, ha = 'right')

plt.savefig('fit')

#plot actual vs. model temp


sns.residplot(temperature, pred, lowess=True)

plt.savefig('residue')

for i in range(len(pred)):
	if (abs(pred[i] - temperature[i])/(temperature[i]*1.0)) > 0.1:
		plt.figure()
		plt.plot(df.loc[i].wavelength, df.loc[i].flux, color = 'k')
		plt.axvline(3980)
		plt.axvline(4920)
		plt.axvline(5070)
		plt.axvline(5950)
		plt.savefig('spectra' + str(i))

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
