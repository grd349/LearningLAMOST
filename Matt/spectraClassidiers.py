#!/usr/bin/env python3

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors

import time

sfile = '/data2/mrs493/my_data2.csv'

df = pd.read_csv(sfile, sep=',')

colour = sp.reshape(df.colour, (-1, 1))
    #reshape the colour to a column vector for use in the algorithm
    
designation = sp.array(df.designation.tolist())

tempf = sp.array(df.teff.tolist())
tempi = sp.array([int(i) for i in tempf])

fig, ax = plt.subplots()
ax.scatter(colour, tempf)
ax.set_xlabel('Colour Feature / B - V')
ax.set_ylabel('Temperature \ K')
ax.set_title('Colour Feature vs. Temperature')
plt.show()
    #plot the colour feature of each curve against its temperature

"""
possibly remove SVC, takes long time (~4 mins per fold)
"""

names = ['KNeighbours', 'RadiusNeighbours', 'Random Forest Regressor', 'Random Forest Classifier', 'SGD', 'SVC']
classifiers = [neighbors.KNeighborsRegressor(), neighbors.RadiusNeighborsRegressor(),
               RandomForestRegressor(), RandomForestClassifier(), linear_model.SGDClassifier(), svm.SVC()]
regressors = [1, 1, 1, 0, 0, 0]
    #load the random forest clssifier

kf = cross_validation.KFold(n = len(colour), n_folds = 10, shuffle = True)
    #use kfolds to split the data

for name, clf, regressor in zip(names, classifiers, regressors):

    if regressor: temp = tempf
    else: temp = tempi
    
    models = []
    importance = []
    
    print 'Time per fold:'
    
    for train_index, test_index in kf:
        t0 = time.time()
        
        #cycle through each kfold and use it as a training set for the algorithm, using the remaining folds as test sets
        X_train, X_test = colour[train_index], colour[test_index]
        y_train, y_test = temp[train_index], temp[test_index]
        desig_train, desig_test = designation[train_index], designation[test_index]
            #split the data into the given folds (need data in an sp.array for indexing to work)
        clf = clf.fit(X_train, y_train)
            #fit the model the the current training set
    
        test_pred = clf.predict(X_test)
            #Use the model to predict the temperatures of the test set
        
        models.append(clf.predict(colour))
        
        #importance.append(clf.feature_importances_)
        
        t1 = time.time()
        
        print '{:.02f} s'.format(t1-t0)
    
    final = sp.mean(models, 0)
    
    fig, ax = plt.subplots(2,2)
    
    fig.suptitle(name)
        
    ax[0][0].scatter(temp, final)
    ax[0][0].set_xlabel('Actual temperature \ K')
    ax[0][0].set_ylabel('Predicted temperature \ K')
    ax[0][0].set_title('Actual vs. Predicted temperature')
        #plot the actual vs. predicted temperature
    
    error = final - temp
        #calculate the error of the fit
    
    MAD = stats.mad_std(error)
        #calculate the MAD of the data
    
    sns.kdeplot(error, ax=ax[0][1], shade=True)
    ax[0][1].set_xlabel('Absolute Error')
    ax[0][1].set_ylabel('Fraction of Points with\nGiven Error')
    ax[0][1].set_title('KDE of Absolute Error\non Temperature Prediction')
        #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])
        
    sns.residplot(temp, final, lowess = True, ax = ax[1][0], line_kws={'color': 'red'})
    ax[1][0].set_title('Residuals of Prediction')
    ax[1][0].set_xlabel('Actual Temperature \ K')
    ax[1][0].set_ylabel('Prediction Residual \ K')
        #plot the residuals of the predicted temperatures
    ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction')
        #write the MAD in the plot
        
        """
        look into residplot - appears residual not propotrional to error (see SGD plot)
        """
        
    test_index = sp.argmax(abs(error))
    df_index = df.loc[df.designation==designation[test_index]].index[0]
        
    spectrum = Spectrum('/data2/mrs493/DR1/' + df.get_value(df_index,'filename'))
    spectrum.plotFlux(ax = ax[1][1], Tpred = final[test_index], Teff = temp[test_index])
    
    ax[1][1].set_xlabel('Wavelength \ Angstroms')
    ax[1][1].set_ylabel('Flux')
    ax[1][1].set_title('Spectra and model blackbody curve\nfor greatest outlier')
        
    plt.tight_layout()
    
    plt.show()
    
    spectrum.plotFlux(Tpred = final[test_index], Teff = temp[test_index])
    
    plt.show()
    
    #print sp.mean(importance, 0)
    
    '''
    add more classifiers
    optimise hyperparameters
    add other features i.e. change band widths, move position, use single band etc.
    find feature importance
    compare cross-validatted to non-cross-validated models
    '''
    
    
        
    
