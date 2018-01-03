#!/usr/bin/env python3

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor

def getFeatures(df):
    B = sp.array(df["B"].tolist())
    R = sp.array(df["R"].tolist())
    I = sp.array(df["I"].tolist())
    V = sp.array(df["V"].tolist())
    Ha = sp.array(df["Ha"].tolist())
    Hb = sp.array(df["Hb"].tolist())
    Hg = sp.array(df["Hg"].tolist())
    totCounts = sp.array(df["totalCounts"].tolist())
    randomFeature = sp.random.normal(0.5,0.2,len(totCounts))
    return sp.column_stack((B-V,B-R,B-I,V-R,V-I,R-I,totCounts,Ha,Hb,Hg,randomFeature))

#i = 0
    
sfile = 'temp2.csv'    ###filename###

df = pd.read_csv(sfile, sep=',')

#
imputer = Imputer()
#

train, test = train_test_split(df, test_size=0.2)

X_train = getFeatures(train)
X_test = getFeatures(test)

#
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
#

y_train = train.teff.tolist()
y_test = test.teff.tolist()

clf = RandomForestRegressor(n_estimators=80,max_depth=10)

clf = clf.fit(X_train, y_train)
    #fit the model the the current training set

final = clf.predict(X_test)
    #Use the model to predict the temperatures of the test set
        
fig, ax = plt.subplots(2,2)

fig.suptitle('Random Forest Regressor')
    
ax[0][0].scatter(y_test, final)
ax[0][0].set_xlabel('Actual temperature \ K')
ax[0][0].set_ylabel('Predicted temperature \ K')
ax[0][0].set_title('Actual vs. Predicted temperature')
    #plot the actual vs. predicted temperature

error = final - y_test
    #calculate the error of the fit

MAD = stats.mad_std(error)
    #calculate the MAD of the data

sns.kdeplot(error, ax=ax[0][1], shade=True)
ax[0][1].set_xlabel('Absolute Error')
ax[0][1].set_ylabel('Fraction of Points with\nGiven Error')
ax[0][1].set_title('KDE of Absolute Error\non Temperature Prediction')
    #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])
    
sns.residplot(sp.array(y_test), final, lowess = True, ax = ax[1][0], line_kws={'color': 'red'})
ax[1][0].set_title('Residuals of Prediction')
ax[1][0].set_xlabel('Actual Temperature \ K')
ax[1][0].set_ylabel('Prediction Residual \ K')
    #plot the residuals of the predicted temperatures
ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
    #write the MAD in the plot
    
"""
look into residplot - appears residual not propotrional to error (see SGD plot)
"""
  
test_index = sp.argmax(abs(error))

spectrum = Spectrum('/data2/mrs493/DR1_3/' + test.filename.tolist()[test_index])    ###filename###
spectrum.plotFlux(ax = ax[1][1], Tpred = final[test_index], Teff = y_test[test_index])


ax[1][1].set_xlabel('Wavelength \ Angstroms')
ax[1][1].set_ylabel('Flux')
ax[1][1].set_title('Spectra and model blackbody curve\nfor greatest outlier')
ax[1][1].legend()
    
plt.tight_layout()

plt.show()

ft = ['BV', 'BR', 'BI', 'VR', 'VI', 'RI', 'totC', 'Ha', 'Hb', 'Hg', 'rand']
imp = clf.feature_importances_

print 'Feature\tImportance\t\tComp to rand'

for f,i in zip(ft, imp):
    print'{}  \t{}   \t{}'.format(f, i, i-imp[-1])
    
'''
add more classifiers
optimise hyperparameters
add other features i.e. change band widths, move position, use single band etc.
find feature importance
compare cross-validatted to non-cross-validated models
'''
