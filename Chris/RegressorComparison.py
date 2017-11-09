import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from astropy.stats import mad_std
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

#Reads in dataframe
sfile = 'spectra_dataframe.csv'
df = pd.read_csv(sfile, sep=',')

#Reads in temperature and features from dataframe
BV = np.array(df["BV"].tolist())
BR = np.array(df["BR"].tolist())
BI = np.array(df["BI"].tolist())
VR = np.array(df["VR"].tolist())
VI = np.array(df["VI"].tolist())
RI = np.array(df["RI"].tolist())

totCounts = np.array(df["totCounts"].tolist())
randomFeature = np.random.normal(0.5,0.2,len(totCounts))
temps = np.array(df["teff"].tolist())
desig = np.array(df["designation"].tolist())

features = np.column_stack((BV,BR,BI,VR,VI,RI,totCounts,randomFeature))
    
features_train, features_test, temp_train, temp_test = train_test_split(features, temps, test_size=0.1)

names = ["Random Forest", "Ada Boost", "Huber", "Linear Regression", "K Neighbours", "RANSAC", "TheilSen", "Quadratic", "Cubic"]

classifiers = [RandomForestRegressor(),
               AdaBoostRegressor(),
               HuberRegressor(),
               LinearRegression(),
               KNeighborsRegressor(),
               RANSACRegressor(),
               TheilSenRegressor(),
               PolynomialFeatures(degree=2),
               PolynomialFeatures(degree=3)]


fig, axes = plt.subplots(3,3,sharex=True,sharey=True)
fig.suptitle('Regressor Comparison',y=0.98,fontsize=18)
fig.text(0.5, 0.04, 'Actual Temperature / K', ha='center')
fig.text(0.04, 0.5, 'Predicted Temperature / K', va='center', rotation='vertical')

for i,ax in enumerate(axes.flatten()):
    
    classifiers[i].fit(features_train,temp_train)
    
    test_pred = classifiers[i].predict(features_test)
    
    error = test_pred - temp_test

    MAD = mad_std(error)

    ax.scatter(temp_test, classifiers[i].predict(features_test))
    ax.set_title(names[i])
    ax.set_ylim([3000,9000])
    ax.annotate('{0:.2f}'.format(MAD), xy=(0.63, 0.05), xycoords='axes fraction',color='r')
    
    #plt.tight_layout()

plt.show()