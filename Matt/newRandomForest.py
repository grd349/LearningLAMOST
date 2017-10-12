from astropy.io import fits
import scipy as sp
import matplotlib.pyplot as plt
import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

from fits import Spectra

def blackbody(T):
    wavelength = sp.linspace(3000, 9000, 3001)
    h = 6.63e-34
    c = 3e8
    k = 1.38e-23
    return (8*sp.pi*h*c)/((wavelength*1e-10)**5*(sp.exp(h*c/((wavelength*1e-10)*k*T))-1))

temp = sp.array(sp.random.normal(6000, 1000, 200))

colour = []

for t in temp:
    spectra = blackbody(t)
    B = sp.sum(spectra[490:960])
    V = sp.sum(spectra[1035:1475])
    colour.append(sp.log10(V/B))

colour = sp.reshape(colour, (-1, 1))

for i in range(len(temp)):
    temp[i] = int(temp[i])

fig, ax = plt.subplots()
ax.scatter(colour, temp)
ax.set_xlabel('Colour / log(V/B)')
ax.set_ylabel('Temperature \ K')
ax.set_title('Colour vs. Temperature ')

clf = RandomForestClassifier()
kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)

"""
Need to add in cross validation method
"""

clf.fit(colour, temp)

pred = clf.predict(colour)

error = pred - temp

fig, ax = plt.subplots()
ax.scatter(temp, pred)
ax.set_xlabel('Actual temperature \ K')
ax.set_ylabel('Predicted temperature \ K')
ax.set_title('Actual vs. Predicted temperature')

fig, ax = plt.subplots()
ax.hist(error)
ax.set_title('Error of Prediction')

"""
spectra = Spectra('../Data/DR1/*.fits')
    
colour = sp.reshape(spectra.colour, (-1, 1)) #input data needs each sample in a separate row, not column
                   
temperature = sp.array([int(i.flux[100]) for i in spectra.spectra]) #need to find actual temperatures, currently using flux as a placeholder
                      
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
    
clf.fit(colour, temperature)

plt.scatter(colour, clf.predict(colour) - temperature)

plt.show()
"""