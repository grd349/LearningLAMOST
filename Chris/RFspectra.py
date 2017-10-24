import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.stats import mad_std

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

clf = RandomForestClassifier()

#Reads in dataframe
sfile = 'spectra_dataframe.csv'
df = pd.read_csv(sfile, sep=',')

#Reads in temperature and B-V from dataframe
colour = np.reshape(df.feature, (-1, 1))

temps = np.array(df["teff"].tolist())
desig = np.array(df["designation"].tolist())

for i in range(len(temps)):
    temps[i] = int(temps[i])

kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)
j=1

#Uses k-folds to split the data into 5 sets and performs training on 4/5 sets then testing on the 5th set
#in all five ways
for train_index, test_index in kf: 
    X_train, X_test = colour[train_index], colour[test_index]
    y_train, y_test = temps[train_index], temps[test_index]
    desig_train, desig_test = desig[train_index], desig[test_index]
    
    #Fits the random forest to the training set and then predicts the temperature of the test set
    clf = clf.fit(X_train,y_train)
    test_pred = clf.predict(X_test)
    
    #Calculates absolute error on each point
    error = test_pred - y_test
    
    #Calculates the median absolute deviation
    #MAD = np.median([abs(i) for i in(test_pred - np.median(test_pred))])
    MAD = mad_std(error)
    
    #alpha=abs(error)/np.amax(abs(error))
    
    fig, ax = plt.subplots(2,2)
    
    #Plots machine learned temperature against actual temperature of the spectra
    ax[0][0].scatter(y_test, test_pred)
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
    sns.residplot(y_test, test_pred, lowess=True, ax=ax[1][0])
    ax[1][0].set_xlabel('Actual Temperature / K')
    ax[1][0].set_ylabel('Residual of Fit')
    ax[1][0].set_title('Residual Plot')
    
    alpha = abs(error)/np.amax(abs(error))
    col = y_test/np.amax(y_test)
    colArray = np.asarray([(col[i], 0, 0.5, alpha[i]) for i in range(len(alpha))])
    
    ax[1][1].scatter(y_test, test_pred, c=colArray)
    ax[1][1].set_xlabel('Actual Temperature / K')
    ax[1][1].set_ylabel('Predicted Temperature / K')
    ax[1][1].set_title('Predicted vs. Actual Temp using RFC')
    ax[1][1].set_ylim([3500,9000])
    
    index = np.argmax(abs(error))
    row_index = df.loc[df['designation']==desig_test[index]].index[0]
    flux = df.get_value(row_index,"flux")
    wavelength = df.get_value(row_index,"wavelength")
    
    print(flux[0])
    
    for i in range(len(flux)):
        flux[i] = float(flux[i])
        wavelength[i] = float(wavelength[i])
    
    fig, ax2 = plt.subplots()
    ax2.plot(flux, wavelength)
    
    #Adds MAD value as text in the bottom right of figure
    #ax[1][0].text(,0,'MAD = ' + str(MAD))
    ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy=(0.05, 0.90), xycoords='axes fraction')
    
    filename = "RFKfolds" + str(j)
    j+=1
    
    plt.tight_layout()
    #plt.savefig(filename)
    
plt.show()
    
"""
#Calculates the mean and standard deviation from accuracy list
mean_accuracy = np.mean(accuracy)
std_accuracy = np.std(accuracy)

clf.fit(colour, temps)
pred = clf.predict(colour)

error = pred - temps

#Predicts temperature for all points and plots this against true temperatures
fig, ax1 = plt.subplots()
ax1.scatter(temps, pred)
ax1.set_xlabel('Actual Temperature / K')
ax1.set_ylabel('Predicted Temperature / K')
ax1.set_title('Prediction of Spectral Temperature using Random Forest Classifier')
ax1.text(9000, 4000, '{} +/- {}'.format(round(mean_accuracy,3), round(std_accuracy,3)), size = 15, ha = 'right')
plt.savefig('RFspectraOverfit')
#plt.show()

fig, ax2 = plt.subplots()
sns.kdeplot(error, ax=ax2, shade=True)
ax2.set_xlabel('Absolute Error / K')
ax2.set_ylabel('Fraction of Points with Given Error')
ax2.set_title('Kernel Density Estimator for Absolute Errors on Random Forest Model')
plt.savefig('RFspectraOverfitkde')
#plt.show()

fig, ax3 = plt.subplots()
sns.residplot(temps, pred, lowess=True, ax=ax3)
ax3.set_xlabel('Actual Temperature / K')
ax3.set_ylabel('Residual of Fit')
ax3.set_title('Residual Plot for Predicted vs. Actual Spectral Temperature')
plt.savefig('RFspectraOverfitresid')
#plt.show()
"""
