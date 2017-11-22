#!/usr/bin/env python3

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor


sfile = '/data2/mrs493/my_data2.csv'

df = pd.read_csv(sfile, sep=',')

colour = sp.reshape(df.colour, (-1, 1))
    #reshape the colour to a column vector for use in the algorithm
    x
designation = sp.array(df.designation.tolist())

temp = sp.array(df.teff.tolist())

names = ['KNeighbours', 'Radius Neighbours', 'Random Forest Regressor',
             'Linear Regression', 'Gaussian Process Regressor', 'Ada Boost Classifier']
classifiers = [KNeighborsRegressor(), RadiusNeighborsRegressor(), RandomForestRegressor(),
               LinearRegression(), GaussianProcessRegressor(), AdaBoostRegressor()]

train, test = train_test_split(df, test_size=0.2)

X_train = sp.reshape(train.colour.tolist(), (-1, 1))
X_test = sp.reshape(test.colour.tolist(), (-1, 1))

y_train = train.teff.tolist()
y_test = test.teff.tolist()

final = []
MAD = []

for name, clf in zip(names, classifiers):
    
    clf = clf.fit(X_train, y_train)
        #fit the model the the current training set

    final.append(clf.predict(X_test))
        #Use the model to predict the temperatures of the test set
            
    fig, ax = plt.subplots(2,2)
    
    fig.suptitle(name)
        
    ax[0][0].scatter(y_test, final[-1])
    ax[0][0].set_xlabel('Actual temperature \ K')
    ax[0][0].set_ylabel('Predicted temperature \ K')
    ax[0][0].set_title('Actual vs. Predicted temperature')
        #plot the actual vs. predicted temperature
    
    error = final[-1] - y_test
        #calculate the error of the fit
    
    MAD.append(stats.mad_std(error))
        #calculate the MAD of the data
    
    sns.kdeplot(error, ax=ax[0][1], shade=True)
    ax[0][1].set_xlabel('Absolute Error')
    ax[0][1].set_ylabel('Fraction of Points with\nGiven Error')
    ax[0][1].set_title('KDE of Absolute Error\non Temperature Prediction')
        #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])
        
    sns.residplot(sp.array(y_test), final[-1], lowess = True, ax = ax[1][0], line_kws={'color': 'red'})
    ax[1][0].set_title('Residuals of Prediction')
    ax[1][0].set_xlabel('Actual Temperature \ K')
    ax[1][0].set_ylabel('Prediction Residual \ K')
        #plot the residuals of the predicted temperatures
    ax[1][0].annotate('MAD = {0:.2f}'.format(MAD[-1]), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
        #write the MAD in the plot
        
    """
    look into residplot - appears residual not propotrional to error (see SGD plot)
    """
        
    test_index = sp.argmax(abs(error))

    spectrum = Spectrum('/data2/mrs493/DR1/' + test.filename.tolist()[test_index])
    spectrum.plotFlux(ax = ax[1][1], Tpred = final[-1][test_index], Teff = y_test[test_index])
    
    ax[1][1].set_xlabel('Wavelength \ Angstroms')
    ax[1][1].set_ylabel('Flux')
    ax[1][1].set_title('Spectra and model blackbody curve\nfor greatest outlier')
    ax[1][1].legend()
        
    plt.tight_layout()
    
    plt.show()
    
    #spectrum.plotFlux(Tpred = final[-1][index], Teff = temp[index])
    #plt.show()
    
'''
add more classifiers
optimise hyperparameters
add other features i.e. change band widths, move position, use single band etc.
find feature importance
compare cross-validatted to non-cross-validated models
'''

fig, ax = plt.subplots(3, 3, sharex = True, sharey = True)
ax = sp.ndarray.flatten(ax)
fig.suptitle('Regressor Comparison')

for i in range(len(ax)):
    ax[i].scatter(y_test, final[i])
    ax[i].set_title(names[i])
    ax[i].annotate('MAD = {0:.2f}'.format(MAD[i]), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
    
