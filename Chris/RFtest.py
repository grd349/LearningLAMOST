"""
from astropy.io import fits

import glob
from readFits import Spectra
"""
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def blackbody(T):
    h = 6.63e-34
    c = 3e8
    k = 1.38e-23
    wavelength = np.linspace(3000,9000,4000)
    E = 1e-4*(8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))
    
    letters = {"B":[3980,4920], "V":[5070,5950]}
    bandCounts = {"B":0, "V":0}
        
    for letter in letters:
        lower = np.searchsorted(wavelength,letters[letter][0],side="left")
        upper = np.searchsorted(wavelength,letters[letter][1],side="right")       
        bandCounts[letter] = np.sum(E[lower:upper])
        
    return np.log10(bandCounts["V"])-np.log10(bandCounts["B"])

temps = np.random.normal(6000,1000,200)
for i in range(len(temps)):
    temps[i] = int(temps[i])

blackbodyList = []

for i in temps:
    blackbodyList.append(blackbody(i))

fig, ax1 = plt.subplots()
ax1.scatter(blackbodyList,temps)
ax1.set_xlabel('V-B Feature')
ax1.set_ylabel('Temperature / K')
ax1.set_title('Plot of Temp vs. V-B Feature for Modelled Blackbody Spectra')
plt.show()    
    

#spectra = Spectra('../Data/DR1/*.fits')
    
colour = np.reshape(blackbodyList, (-1, 1))
                   
#temperature = np.array([int(i.flux[100]) for i in spectra.spectra])
                      
   
clf = RandomForestClassifier()

kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)

error = 0

for train_index, test_index in kf:   
    X_train, X_test = colour[train_index], colour[test_index]
    y_train, y_test = temps[train_index], temps[test_index]
    clf = clf.fit(X_train,y_train)
    test_pred = clf.predict(X_test)
    
    error = test_pred - y_test
    fig, ax2 = plt.subplots()
    ax2.scatter(y_test, test_pred)
    ax2.set_xlabel('Actual Temperature / K')
    ax2.set_ylabel('Predicted Temperature / K')
    ax2.set_title('Plot of Predicted vs. Actual Temperature for Modelled Blackbodies')
    plt.show()
    
    fig, ax3 = plt.subplots()
    ax3.hist(error)
    ax3.set_xlabel('Actual Temperature / K')
    ax3.set_ylabel('Predicted Temperature / K')
    ax3.set_title('Plot of Predicted vs. Actual Temperature for Modelled Blackbodies')
    plt.show()
    
"""
clf.fit(colour, temps)
pred = clf.predict(colour)

error = pred - temps

fig, ax2 = plt.subplots()
ax2.scatter(temps, clf.predict(colour))
ax2.set_xlabel('Actual Temperature / K')
ax2.set_ylabel('Predicted Temperature / K')
ax2.set_title('Plot of Predicted vs. Actual Temperature for Modelled Blackbodies')
plt.show()

fig, ax3 = plt.subplots()
ax3.hist(error)
ax3.set_xlabel('Actual Temperature / K')
ax3.set_ylabel('Predicted Temperature / K')
ax3.set_title('Plot of Predicted vs. Actual Temperature for Modelled Blackbodies')
plt.show()
"""