#!/usr/bin/env python3

#change to use new csv, also need to merge with catalog while running

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

cfile = '/data2/cpb405/dr1_stellar.csv'
catalog = pd.read_csv(cfile, sep='|')
catalog.drop_duplicates(subset = 'designation', inplace = True)
    
sfile = 'spectra.csv'    ###filename###

spec = pd.read_csv(sfile, sep=',')

df = catalog.merge(spec, on='designation', how='inner')

#
features = sp.array(df.columns[39:])
colours = features[sp.array([feat != 'cTotal' and feat[0]=='c' for feat in features])]

print 1

for idx in range(len(colours)-1):
    for j in sp.arange(idx+1, len(colours)):
        df[colours[idx] + '-' + colours[j]] = df.loc[:,colours[idx]] - df.loc[:,colours[j]]
        
print 2
        
features = sp.array(df.columns[39:])

#
imputer = Imputer(missing_values = 0)
df[features] = imputer.fit_transform(df[features])

print 3

train, test = train_test_split(df, test_size=0.2)

X_train = train[features]
X_test = test[features]

#X_train = imputer.transform(X_train)
#X_test = imputer.transform(X_test)

y_train = train['teff'].tolist()
y_test = test['teff'].tolist()

ends = [sp.amin(y_test), sp.amax(y_test)]

print 4

#
parameter_grid = [{'n_estimators':[1,10,20,40,60,80,100],'max_depth':[1,10,100,1000]}]
regr = GridSearchCV(RandomForestRegressor(), parameter_grid, cv=5)
regr.fit(df[features], df.teff.tolist())
print regr.best_params_
#

clf = RandomForestRegressor(n_estimators=regr.best_params_['n_estimators'],max_depth=regr.best_params_['max_depth'])

clf = clf.fit(X_train, y_train)
    #fit the model the the current training set

final = clf.predict(X_test)
    #Use the model to predict the temperatures of the test set

fig, ax = plt.subplots(2,2)

fig.suptitle('Random Forest Regressor')
    
ax[0][0].scatter(y_test, final)
ax[0][0].plot(ends, ends, ls = ':', color = 'red')
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
ax[0][1].set_ylabel('Fraction of Points with Error')
ax[0][1].set_title('KDE of Absolute Error')
    #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])
    
sns.residplot(sp.array(y_test), final, lowess = True, ax = ax[1][0], line_kws={'color': 'red'})
ax[1][0].set_title('Residuals of Prediction')
ax[1][0].set_xlabel('Actual Temperature \ K')
ax[1][0].set_ylabel('Prediction Residual \ K')
    #plot the residuals of the predicted temperatures
ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
    #write the MAD in the plot
  
test_index = sp.argmax(abs(error))

spectrum = Spectrum('/data2/mrs493/DR1_3/' + test.filename.tolist()[test_index])    ###filename###
spectrum.plotFlux(ax = ax[1][1], Tpred = final[test_index], Teff = y_test[test_index], label = 'Outlier')


ax[1][1].set_xlabel('Wavelength \ Angstroms')
ax[1][1].set_ylabel('Flux')
ax[1][1].set_title('Spectra of Greatest Outlier')
ax[1][1].legend()
    
plt.tight_layout()
plt.savefig('figures/tempModel.pdf')
plt.show()

imp = clf.feature_importances_

fig, ax = plt.subplots()
sns.barplot(features, imp)
ax.set_xlabel('Features')
ax.set_ylabel('% Importance')
ax.set_title('Feature Importance')
plt.show()

