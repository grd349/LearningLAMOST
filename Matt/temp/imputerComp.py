#!/usr/bin/env python3

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

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
    
sfile = 'temp2.csv'    ###filename###

df = pd.read_csv(sfile, sep=',')

imputer = Imputer()

train, test = train_test_split(df, test_size=0.2)

X_train = getFeatures(train)
X_test = getFeatures(test)

#
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
#

y_train = train.teff.tolist()
y_test = test.teff.tolist()

names = ['KNeighbours', 'Radius Neighbors', 'Random Forest', 'Linear Regression', 'Gaussian Process', 'Ada Boost', 'Huber', 'RANSAC', 'Theil-Sen', ]
classifiers = [KNeighborsRegressor(), RadiusNeighborsRegressor(), RandomForestRegressor(), LinearRegression(), GaussianProcessRegressor(), AdaBoostRegressor(), HuberRegressor(), RANSACRegressor(), TheilSenRegressor()]
#theilSen, SVR
fig, ax = plt.subplots(nrows = 3, ncols = 3, sharex = True)

ax.flat[6].set_xlabel('LAMOST')
ax.flat[6].set_ylabel('Model')

for i in range(len(classifiers)):
    
    clf = classifiers[i]
    
    clf = clf.fit(X_train, y_train)
        #fit the model the the current training set
    
    final = clf.predict(X_test)
        #Use the model to predict the temperatures of the test set
    
    fig.suptitle('Regressor Comparison')
    
    error = final - y_test
        #calculate the error of the fit
    
    MAD = stats.mad_std(error)

    ends = [sp.amin(y_test), sp.amax(y_test)]

    ax.flat[i].scatter(y_test, final)
    ax.flat[i].plot(ends, ends, ls = ':', color = 'red')
    ax.flat[i].set_title(names[i])
    ax.flat[i].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'red')
    
plt.savefig('comp.pdf')