#!/usr/bin/env python3

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

def getColourFeatures(df):
    BV = sp.array(df["BV"].tolist())
    BR = sp.array(df["BR"].tolist())
    BI = sp.array(df["BI"].tolist())
    VR = sp.array(df["VR"].tolist())
    VI = sp.array(df["VI"].tolist())
    RI = sp.array(df["RI"].tolist())
    return sp.column_stack((BV,BR,BI,VR,VI,RI))

def getLineFeatures(df):
    Ha = sp.array(df["Ha"].tolist())
    Hb = sp.array(df["Hb"].tolist())
    Hg = sp.array(df["Hg"].tolist())
    return sp.column_stack((Ha, Hb, Hg))
    
sfile = '/data2/mrs493/my_data2.csv'

df = pd.read_csv(sfile, sep=',')

#a = getFeatures(df)

train, test = train_test_split(df, test_size=0.2)

Xc_train = getColourFeatures(train)
Xc_test = getColourFeatures(test)

Xl_train = getLineFeatures(train)
Xl_test = getLineFeatures(test)

y_train = train.teff.tolist()
y_test = test.teff.tolist()



clfc = RandomForestRegressor(n_estimators=80,max_depth=10)

clfc = clfc.fit(Xc_train, y_train)
    #fit the model the the current training set

finalc = clfc.predict(Xc_test)
    #Use the model to predict the temperatures of the test set
        
fig1, ax1 = plt.subplots(2,2)

fig1.suptitle('Random Forest Regressor\nColour Features')
    
ax1[0][0].scatter(y_test, finalc)
ax1[0][0].set_xlabel('Actual temperature \ K')
ax1[0][0].set_ylabel('Predicted temperature \ K')
ax1[0][0].set_title('Actual vs. Predicted temperature')
    #plot the actual vs. predicted temperature

errorc = finalc - y_test
    #calculate the error of the fit

MADc = stats.mad_std(errorc)
    #calculate the MAD of the data

sns.kdeplot(errorc, ax=ax1[0][1], shade=True)
ax1[0][1].set_xlabel('Absolute Error')
ax1[0][1].set_ylabel('Fraction of Points with\nGiven Error')
ax1[0][1].set_title('KDE of Absolute Error\non Temperature Prediction')
    #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])
    
sns.residplot(sp.array(y_test), finalc, lowess = True, ax = ax1[1][0], line_kws={'color': 'red'})
ax1[1][0].set_title('Residuals of Prediction')
ax1[1][0].set_xlabel('Actual Temperature \ K')
ax1[1][0].set_ylabel('Prediction Residual \ K')
    #plot the residuals of the predicted temperatures
ax1[1][0].annotate('MAD = {0:.2f}'.format(MADc), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
    #write the MAD in the plot
    
"""
look into residplot - appears residual not propotrional to error (see SGD plot)
"""
    
test_index = sp.argmax(abs(errorc))

spectrum = Spectrum('/data2/mrs493/DR1/' + test.filename.tolist()[test_index])
spectrum.plotFlux(ax = ax1[1][1], Tpred = finalc[test_index], Teff = y_test[test_index])

ax1[1][1].set_xlabel('Wavelength \ Angstroms')
ax1[1][1].set_ylabel('Flux')
ax1[1][1].set_title('Spectra and model blackbody curve\nfor greatest outlier')
ax1[1][1].legend()
    
plt.tight_layout()

plt.show()

ft = ['BV', 'BR', 'BI', 'VR', 'VI', 'RI']
imp = clfc.feature_importances_

print 'Feature\tImportance'

for f,i in zip(ft, imp):
    print'{}  \t{}'.format(f, i)

    
    
clfl = RandomForestRegressor(n_estimators=80,max_depth=10)

clfl = clfl.fit(Xl_train, y_train)
    #fit the model the the current training set

finall = clfl.predict(Xl_test)
    #Use the model to predict the temperatures of the test set
        
fig2, ax2 = plt.subplots(2,2)

fig2.suptitle('Random Forest Regressor\nLine Features')
    
ax2[0][0].scatter(y_test, finall)
ax2[0][0].set_xlabel('Actual temperature \ K')
ax2[0][0].set_ylabel('Predicted temperature \ K')
ax2[0][0].set_title('Actual vs. Predicted temperature')
    #plot the actual vs. predicted temperature

errorl = finall - y_test
    #calculate the error of the fit

MADl = stats.mad_std(errorl)
    #calculate the MAD of the data

sns.kdeplot(errorl, ax=ax2[0][1], shade=True)
ax2[0][1].set_xlabel('Absolute Error')
ax2[0][1].set_ylabel('Fraction of Points with\nGiven Error')
ax2[0][1].set_title('KDE of Absolute Error\non Temperature Prediction')
    #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])
    
sns.residplot(sp.array(y_test), finall, lowess = True, ax = ax2[1][0], line_kws={'color': 'red'})
ax2[1][0].set_title('Residuals of Prediction')
ax2[1][0].set_xlabel('Actual Temperature \ K')
ax2[1][0].set_ylabel('Prediction Residual \ K')
    #plot the residuals of the predicted temperatures
ax2[1][0].annotate('MAD = {0:.2f}'.format(MADl), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
    #write the MAD in the plot
    
"""
look into residplot - appears residual not propotrional to error (see SGD plot)
"""
    
test_index = sp.argmax(abs(errorl))

spectrum = Spectrum('/data2/mrs493/DR1/' + test.filename.tolist()[test_index])
spectrum.plotFlux(ax = ax2[1][1], Tpred = finall[test_index], Teff = y_test[test_index])

ax2[1][1].set_xlabel('Wavelength \ Angstroms')
ax2[1][1].set_ylabel('Flux')
ax2[1][1].set_title('Spectra and model blackbody curve\nfor greatest outlier')
ax2[1][1].legend()
    
plt.tight_layout()

plt.show()

ft = ['Ha', 'Hb', 'Hg']
imp = clfl.feature_importances_

print 'Feature\tImportance'

for f,i in zip(ft, imp):
    print'{}  \t{}'.format(f, i)

fig3, ax3 = plt.subplots()

ax3.scatter(finall, finalc)

allowance = 1000

for i in range(len(finall)):
    if abs(finall[i] - finalc[i])>allowance:
        spectrum = Spectrum('/data2/mrs493/DR1/' + test.filename.tolist()[i])
        spectrum.plotFlux(Tpred = finall[i], Teff = finalc[i])
        plt.legend(['Flux','Line Model', 'Colour Model'])
        plt.xlabel('Wavelength \ Angstroms')
        plt.ylabel('Flux')

'''
add more classifiers
optimise hyperparameters
add other features i.e. change band widths, move position, use single band etc.
find feature importance
compare cross-validatted to non-cross-validated models
'''
    
