#!/usr/bin/env python3

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

sfile = '/data2/mrs493/my_data2.csv'

df = pd.read_csv(sfile, sep=',')

colour = sp.reshape(df.colour, (-1, 1))
    #reshape the colour to a column vector for use in the algorithm
    
designation = sp.array(df.designation.tolist())

temp = sp.array([int(i) for i in df.teff.tolist()])

fig, ax = plt.subplots()
ax.scatter(colour, temp)#, c = temp)
ax.set_xlabel('Colour Feature / B - V')
ax.set_ylabel('Temperature \ K')
ax.set_title('Colour Feature vs. Temperature')
plt.show()
    #plot the colour feature of each curve against its temperature

'''
change colour map to something more easily seen and read temp dep colour
add colour bar
'''

clf = RandomForestClassifier()
    #load the random forest clssifier

kf = cross_validation.KFold(n = len(colour), n_folds = 5, shuffle = True)
    #use kfolds to split the data

for train_index, test_index in kf:
    #cycle through each kfold and use it as a training set for the algorithm, using the remaining folds as test sets
    X_train, X_test = colour[train_index], colour[test_index]
    y_train, y_test = temp[train_index], temp[test_index]
    desig_train, desig_test = designation[train_index], designation[test_index]
        #split the data into the given folds (need data in an sp.array for indexing to work)
    clf = clf.fit(X_train, y_train)
        #fit the model the the current training set
    test_pred = clf.predict(X_test)
        #Use the model to predict the temperatures of the test set

    fig, ax = plt.subplots(2,2)
    
    ax[0][0].scatter(y_test, test_pred)
    ax[0][0].set_xlabel('Actual temperature \ K')
    ax[0][0].set_ylabel('Predicted temperature \ K')
    ax[0][0].set_title('Actual vs. Predicted temperature')
            #plot the actual vs. predicted temperature

    error = test_pred - y_test
        	#calculate the error of the fit

    MAD = stats.mad_std(error)
        #calculate the MAD of the data

    sns.kdeplot(error, ax=ax[0][1], shade=True)
    ax[0][1].set_xlabel('Absolute Error')
    ax[0][1].set_ylabel('Fraction of Points with\nGiven Error')
    ax[0][1].set_title('KDE of Absolute Error on Temperature Prediction')
            #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])
            
    sns.residplot(y_test, test_pred, lowess = True, ax = ax[1][0])
    ax[1][0].set_title('Residuals of Prediction')
    ax[1][0].set_xlabel('Actual Temperature \ K')
    ax[1][0].set_ylabel('Prediction Residual \ K')
        #plot the residuals of the predicted temperatures
    ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction')
   
    test_index = sp.argmax(abs(error))
    df_index = df.loc[df.designation==desig_test[test_index]].index[0]
    
    spectrum = Spectrum('/data2/mrs493/DR1/' + df.get_value(df_index,'filename'))
    spectrum.plotFlux(ax = ax[1][1], T = test_pred[test_index])
    
    ax[1][1].set_xlabel('Wavelength \ Angstroms')
    ax[1][1].set_ylabel('Flux')
    ax[1][1].set_title('Spectra and model blackbody curve\nfor greatest outlier')
    
    ax[1][1].annotate('Error = {}\nClass: {}'.format(error[test_index], spectrum.CLASS), xy = (0.3, 0.1), xycoords = 'axes fraction')
    
    plt.tight_layout()

plt.show()

'''
try other features i.e. change band widths, move position, use single band etc.
'''