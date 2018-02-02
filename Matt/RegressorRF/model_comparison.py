#!/usr/bin/env python3

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
'''
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
    
sfile = 'Files/spectra2.csv'
df = pd.read_csv(sfile, sep=',')

imputer = Imputer()

train, test = train_test_split(df, test_size=0.2)

X_train = getFeatures(train)
X_test = getFeatures(test)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

y_train = train.teff.tolist()
y_test = test.teff.tolist()
'''

bright = 1000
    #number/fraction of stars to include, starting at brightest (set to False to include all)

cfile = '/data2/cpb405/dr1_stellar.csv'
catalog = pd.read_csv(cfile, sep='|')
catalog.drop_duplicates(subset = 'designation', inplace = True)
    
sfile = 'Files/spectra2.csv'    ###filename###

spec = pd.read_csv(sfile, sep=',')

df = catalog.merge(spec, on='designation', how='inner')

if bright and bright%1==0: df = df.sort_values('total', ascending = False)[:bright]
elif bright and bright <= 1: df = df.sort_values('total', ascending = False)[:int(bright*len(df['designation']))]

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

parameters = ['teff', 'logg', 'feh']

names = ['KNeighbours', 'Radius Neighbors', 'Random Forest', 'Linear Regression', 'Gaussian Process', 'Ada Boost', 'Huber', 'RANSAC', 'Theil-Sen', ]
classifiers = [KNeighborsRegressor(), RadiusNeighborsRegressor(), RandomForestRegressor(), LinearRegression(), GaussianProcessRegressor(), AdaBoostRegressor(), HuberRegressor(), RANSACRegressor(), TheilSenRegressor()]

for parameter in parameters:
    print(parameter)
    y_train = train[parameter].tolist()
    y_test = test[parameter].tolist()

    ends = [sp.amin(y_test), sp.amax(y_test)]
    
    fig, ax = plt.subplots(nrows = 3, ncols = 3, sharex = True)
    
    for i in range(len(classifiers)):
        
        clf = classifiers[i]
        
        clf = clf.fit(X_train, y_train)
            #fit the model the the current training set
        
        final = clf.predict(X_test)
            #Use the model to predict the temperatures of the test set
        
        fig.suptitle(parameter + ' Regressor Comparison')
        
        error = final - y_test
            #calculate the error of the fit
        
        MAD = stats.mad_std(error)
    
        ax.flat[i].scatter(y_test, final)
        ax.flat[i].plot(ends, ends, ls = ':', color = 'red')
        ax.flat[i].set_title(names[i])
        ax.flat[i].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
        
    if bright: plt.savefig('Figures/regressor_' + parameter + '_comparison_' + str(bright) + 'B.pdf')
    else: plt.savefig('Figures/regressor_' + parameter + '_comparison.pdf')
plt.show()
