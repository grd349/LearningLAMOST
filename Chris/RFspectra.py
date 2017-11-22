import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import readFits

from astropy.stats import mad_std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation


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
HalphaEW = np.array(df["HalphaEW"].tolist())
HbetaEW = np.array(df["HbetaEW"].tolist())
HgammaEW = np.array(df["HgammaEW"].tolist())

totCounts = np.array(df["totCounts"].tolist())
#spike = np.array(df["spike"].tolist())
#turningPoints = np.array(df["turningPoints"].tolist())

randomFeature = np.random.normal(0.5,0.2,len(totCounts))
temps = np.array(df["teff"].tolist())
desig = np.array(df["designation"].tolist())

features = np.column_stack((BV,BR,BI,VR,VI,RI,totCounts,HalphaEW,HbetaEW,HgammaEW))

kf = cross_validation.KFold(n=len(BV), n_folds=5, shuffle=True)
j = 1
foldAverage = []

features_train, features_test, temp_train, temp_test, desig_train, desig_test = train_test_split(features, temps, desig, test_size=0.5)

#tuned_params = [{'n_estimators':[1,10,20,40,60,80,100],'max_depth':[1,10,100,1000]}]

#clf = GridSearchCV(RandomForestRegressor(), tuned_params, cv=5)
#clf.fit(features_train, temp_train)
#print(clf.best_params_)

clf = RandomForestRegressor(n_estimators=80,max_depth=10)

#Fits the random forest to the training set and then predicts the temperature of the test set
clf = clf.fit(features_train,temp_train)
test_pred = clf.predict(features_test)

#Calculates absolute error on each point
error = test_pred - temp_test

#Calculates the median absolute deviation
MAD = mad_std(error)

importances = clf.feature_importances_
print(importances)

fig, ax = plt.subplots(2,2)
fig.suptitle('Random Forest Regressor',y=1.03,fontsize=18)

#Plots machine learned temperature against actual temperature of the spectra
ax[0][0].scatter(temp_test, test_pred)
ax[0][0].set_xlabel('Actual Temperature / K')
ax[0][0].set_ylabel('Predicted Temperature / K')
ax[0][0].set_title('Predicted vs. Actual Temperature')

#Plots a kernel density estimator plot for the absolute errors
sns.kdeplot(error, ax=ax[0][1], shade=True)
ax[0][1].set_xlabel('Absolute Error / K')
ax[0][1].set_ylabel('Fraction of Points with Error')
ax[0][1].set_title('KDE Plot for Absolute Errors')

#Plots a residual plot for the predicted temp vs. the actual temperature
sns.residplot(temp_test, test_pred, lowess=True, ax=ax[1][0], line_kws={'color':'red'})
ax[1][0].set_xlabel('Actual Temperature / K')
ax[1][0].set_ylabel('Residual of Fit')
ax[1][0].set_title('Residual of Errors')

index = np.argmax(abs(error))
row_index = df.loc[df['designation']==desig_test[index]].index[0]

finalSpectrum = readFits.Spectrum('/data2/mrs493/DR1/' + df.get_value(row_index,'filename'))  

ax[1][1].plot(finalSpectrum.wavelength, finalSpectrum.flux)
ax[1][1].plot(finalSpectrum.wavelength,readFits.blackbody(df.get_value(row_index,'teff'),finalSpectrum.wavelength,finalSpectrum),'--',c='r',label='True Temp.')
ax[1][1].plot(finalSpectrum.wavelength,readFits.blackbody(test_pred[index],finalSpectrum.wavelength,finalSpectrum),'--',c='b',label='Predicted Temp.')
ax[1][1].set_xlabel('Wavelength / Angstroms')
ax[1][1].set_ylabel('Flux')
ax[1][1].set_title('Spectrum for Greatest Outlier')
ax[1][1].legend()

#Adds MAD value as text in the bottom right of figure
#ax[1][0].text(,0,'MAD = ' + str(MAD))
ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy=(0.05, 0.90), xycoords='axes fraction',color='r')
#ax[1][1].annotate('Type = {}'.format(finalSpectrum.CLASS), xy=(0.55, 0.08), xycoords='axes fraction',color='r')

plt.tight_layout()
plt.show()
