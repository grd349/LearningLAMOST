#!/usr/bin/env python3

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

cfile = '/data2/cpb405/dr1_stellar.csv'
catalog = pd.read_csv(cfile, sep='|')
catalog.drop_duplicates(subset = 'designation', inplace = True)
    
sfile = 'spectra2.csv'    ###filename###

spec = pd.read_csv(sfile, sep=',')

df = catalog.merge(spec, on='designation', how='inner')

#
features = sp.array(df.columns[39:])
colours = features[sp.array([feat[0]=='c' for feat in features])]
lines = features[sp.array([feat[0]=='l' for feat in features])]

imputer = Imputer(missing_values = 0)
df[features] = imputer.fit_transform(df[features])

for idx in range(len(colours)-1):
    for j in sp.arange(idx+1, len(colours)):
        df[colours[idx][1:] + '-' + colours[j][1:]] = df.loc[:,colours[idx]] - df.loc[:,colours[j]]

for idx in range(len(lines)-1):
    for j in sp.arange(idx+1, len(lines)):
        df[lines[idx][1:] + '/' + lines[j][1:]] = df.loc[:,lines[idx]]/df.loc[:,lines[j]]

features = sp.array(df.columns[39:])

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

#

train, test = train_test_split(df, test_size=0.2)

X_train = train[features]
X_test = test[features]

y_train = train['logg'].tolist()
y_test = test['logg'].tolist()

ends = [sp.amin(y_test), sp.amax(y_test)]

#
parameter_grid = [{'n_estimators':[1,10,20,40,60,80,100],'max_depth':[1,10,100,1000, 2000, 3000, 4000, 5000],'max_features':sp.arange(3, len(features))[::3]}]#,len(features)]}]
regr = GridSearchCV(RandomForestRegressor(), parameter_grid, cv=5)
regr.fit(df[features], df.teff.tolist())
hyp = regr.best_params_
print(hyp)
#

clf = RandomForestRegressor(n_estimators=hyp['n_estimators'],max_depth=hyp['max_depth'],max_features = hyp['max_features'])
#clf = RandomForestRegressor()

clf = clf.fit(X_train, y_train)
    #fit the model the the current training set

final = clf.predict(X_test)
    #Use the model to predict the temperatures of the test set

error = final - y_test
    #calculate the error of the fit

MAD = stats.mad_std(error)
    #calculate the MAD of the data

fig, ax = plt.subplots(2,2, figsize=(18,12))

fig.suptitle('Random Forest Regressor')
    
ax[0][0].scatter(y_test, final)
ax[0][0].plot(ends, ends, ls = ':', color = 'red')
ax[0][0].set_xlabel('Actual temperature \ K')
ax[0][0].set_ylabel('Predicted temperature \ K')
ax[0][0].set_title('Actual vs. Predicted temperature')
    #plot the actual vs. predicted temperature
ax[0][0].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
    #write the MAD in the plot

sns.kdeplot(error, ax=ax[0][1], shade=True)
ax[0][1].set_xlabel('Absolute Error')
ax[0][1].set_ylabel('Fraction of Points with Error')
ax[0][1].set_title('KDE of Absolute Error')
    #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])

'''    
sns.residplot(sp.array(y_test), final, lowess = True, ax = ax[1][0], line_kws={'color': 'red'})
ax[1][0].set_title('Residuals of Prediction')
ax[1][0].set_xlabel('Actual Temperature \ K')
ax[1][0].set_ylabel('Prediction Residual \ K')
    #plot the residuals of the predicted temperatures
ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
    #write the MAD in the plot
'''

fea = ['']*len(features)

for f in range(len(features)):
    if features[f][0] == 'c' or features[f][0] == 'l':
        fea[f] = features[f][1:]
    else: fea[f] = features[f]

imp = [[x,y] for x,y in sorted(zip(clf.feature_importances_, fea), reverse = True)]

bp = sns.barplot([i[1] for i in imp][:hyp['max_features']], [i[0]  for i in imp][:hyp['max_features']], ax = ax[1][0])
ax[1][0].set_xlabel('Features')
ax[1][0].set_ylabel('% Importance')
ax[1][0].set_title('Feature Importance')
for tick in ax[1][0].get_xticklabels():
    tick.set_rotation(90)
'''    
for x in range(len(imp)):
    bp.text(x,imp[x][0], '{:.2f}'.format(imp[x][0]), color='black', ha='center')
'''  
test_index = sp.argmax(abs(error))

spectrum = Spectrum('/data2/mrs493/DR1_3/' + test.filename.tolist()[test_index])    ###filename###
spectrum.plotFlux(ax = ax[1][1], Tpred = final[test_index], Teff = y_test[test_index], label = 'Outlier')


ax[1][1].set_xlabel('Wavelength \ Angstroms')
ax[1][1].set_ylabel('Flux')
ax[1][1].set_title('Spectra of Greatest Outlier')
ax[1][1].legend()
    
plt.tight_layout()
plt.savefig('figures/loggModel.pdf')
plt.show()

