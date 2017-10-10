from astropy.io import fits
import scipy as sp
import matplotlib.pyplot as plt
import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

from fits import Spectrum

spectra = []

for fitsName in glob.glob('../Data/DR1/*.fits'):
    spectra.append(Spectrum(fitsName))
    
colour = sp.column_stack([sp.array([i.colour for i in spectra])]) #input data needs each sample in a separate row, not column

temperature = sp.array([int(i.flux[-1]) for i in spectra]) #need to find actual temperatures

#need samples and their 'correct values' in scipy arrays, so can index them using kfolds
    
clf = RandomForestClassifier()
kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)

error = 0

for train_index, test_index in kf:   
    X_train, X_test = colour[train_index], colour[test_index]
    y_train, y_test = temperature[train_index], temperature[test_index]
    clf = clf.fit(X_test,y_test)
    test_pred = clf.predict(X_test)
    error += sp.sum(abs(test_pred-y_test)/len(colour))#need to make more mathematical test of accuracy, placeholder
    
print kf

clf.fit(colour, temperature)

plt.scatter(colour, clf.predict(colour) - temperature)

plt.show()
