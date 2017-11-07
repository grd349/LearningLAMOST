import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.stats import mad_std

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from readFits import Spectrum

#Reads in dataframe
sfile = 'spectra_dataframe.csv'
df = pd.read_csv(sfile, sep=',')

#Reads in temperature and features from dataframe
BminusV = np.array(df["BminusV"].tolist())
BminusR = np.array(df["BminusR"].tolist())
VminusR = np.array(df["VminusR"].tolist())
totCounts = np.array(df["totCounts"].tolist())

features = np.column_stack((BminusV,BminusR,VminusR,totCounts))

temps = np.array(df["teff"].tolist())
desig = np.array(df["designation"].tolist())

for i in range(len(temps)):
    temps[i] = int(temps[i])

kf = cross_validation.KFold(n=len(BminusV), n_folds=5, shuffle=True)
j = 1
foldAverage = []
"""
features_train, features_test, temp_train, temp_test = train_test_split(features, temps, test_size=0.5, random_state=0)

tuned_params = [{'min_samples_leaf':[1,10,100], 'n_estimators':[1,10,100]}]


clf = GridSearchCV(RandomForestClassifier(), tuned_params, cv=5)
clf.fit(features_train, temp_train)
print(clf.best_params_)
"""
clf = RandomForestClassifier()
#Uses k-folds to split the data into 5 sets and performs training on 4/5 sets then testing on the 5th set
#in all five ways
for train_index, test_index in kf: 
    features_train, features_test = features[train_index], features[test_index]
    temp_train, temp_test = temps[train_index], temps[test_index]
    desig_train, desig_test = desig[train_index], desig[test_index]
    
    #Fits the random forest to the training set and then predicts the temperature of the test set
    clf = clf.fit(features_train,temp_train)
    test_pred = clf.predict(features_test)
    
    #Calculates absolute error on each point
    error = test_pred - temp_test
    
    #Calculates the median absolute deviation
    #MAD = np.median([abs(i) for i in(test_pred - np.median(test_pred))])
    MAD = mad_std(error)
    
    #alpha=abs(error)/np.amax(abs(error))
    
    fig, ax = plt.subplots(2,2)
    
    #Plots machine learned temperature against actual temperature of the spectra
    ax[0][0].scatter(temp_test, test_pred)
    ax[0][0].set_xlabel('Actual Temperature / K')
    ax[0][0].set_ylabel('Predicted Temperature / K')
    ax[0][0].set_title('Predicted vs. Actual Temp using RFC')
    ax[0][0].set_ylim([3500,9000])
    
    #Plots a kernel density estimator plot for the absolute errors
    sns.kdeplot(error, ax=ax[0][1], shade=True)
    ax[0][1].set_xlabel('Absolute Error / K')
    ax[0][1].set_ylabel('Fraction of Points with Error')
    ax[0][1].set_title('KDE Plot for Absolute Errors')

    #Plots a residual plot for the predicted temp vs. the actual temperature
    sns.residplot(temp_test, test_pred, lowess=True, ax=ax[1][0])
    ax[1][0].set_xlabel('Actual Temperature / K')
    ax[1][0].set_ylabel('Residual of Fit')
    ax[1][0].set_title('Residual Plot')
    
    index = np.argmax(abs(error))
    row_index = df.loc[df['designation']==desig_test[index]].index[0]
    
    spectrum = Spectrum('/data2/mrs493/DR1/' + df.get_value(index,'filename'))  
    
    ax[1][1].plot(spectrum.wavelength, spectrum.flux)
    ax[1][1].set_xlabel('Wavelength / Angstroms')
    ax[1][1].set_ylabel('Flux')
    ax[1][1].set_title('Spectrum for Greatest Outlier')
    
    #Adds MAD value as text in the bottom right of figure
    #ax[1][0].text(,0,'MAD = ' + str(MAD))
    ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy=(0.05, 0.90), xycoords='axes fraction')
    
    filename = "RFKfolds" + str(j)
    j+=1
    
    plt.tight_layout()
    #plt.savefig(filename)
    
    foldAverage.append(clf.predict(features))
    
    importances = clf.feature_importances_
    print(importances)
    
plt.show()

finalModel = np.mean(foldAverage,0)
finalError = finalModel - temps

finalMAD = mad_std(error)

fig, ax2 = plt.subplots(2,2)
    
#Plots machine learned temperature against actual temperature of the spectra
ax2[0][0].scatter(temps, finalModel)
ax2[0][0].set_xlabel('Actual Temperature / K')
ax2[0][0].set_ylabel('Predicted Temperature / K')
ax2[0][0].set_title('Predicted vs. Actual Temp using RFC')
ax2[0][0].set_ylim([3500,9000])

#Plots a kernel density estimator plot for the absolute errors
sns.kdeplot(error, ax=ax2[0][1], shade=True)
ax2[0][1].set_xlabel('Absolute Error / K')
ax2[0][1].set_ylabel('Fraction of Points with Error')
ax2[0][1].set_title('KDE Plot for Absolute Errors')

#Plots a residual plot for the predicted temp vs. the actual temperature
sns.residplot(temps, finalModel, lowess=True, ax=ax2[1][0])
ax2[1][0].set_xlabel('Actual Temperature / K')
ax2[1][0].set_ylabel('Residual of Fit')
ax2[1][0].set_title('Residual Plot')

index = np.argmax(abs(error))
row_index = df.loc[df['designation']==desig[index]].index[0]

finalSpectrum = Spectrum('/data2/mrs493/DR1/' + df.get_value(index,'filename'))  

ax2[1][1].plot(finalSpectrum.wavelength, finalSpectrum.flux)
ax2[1][1].set_xlabel('Wavelength / Angstroms')
ax2[1][1].set_ylabel('Flux')
ax2[1][1].set_title('Spectrum for Greatest Outlier')

#Adds MAD value as text in the bottom right of figure
#ax[1][0].text(,0,'MAD = ' + str(MAD))
ax2[1][0].annotate('MAD = {0:.2f}'.format(finalMAD), xy=(0.05, 0.90), xycoords='axes fraction')

plt.tight_layout()