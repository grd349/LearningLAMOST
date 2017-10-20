#!/usr/bin/env python3

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

sfile = '/data2/mrs493/my_data.csv'

df = pd.read_csv(sfile, sep=',')

colour = sp.reshape(df.colour, (-1, 1))
	#reshape the colour to a column vector for use in the algorithm

temp = sp.array([int(i) for i in df.teff])

fig, ax = plt.subplots()
ax.scatter(colour, temp)#, c = temp)
ax.set_xlabel('Colour Feature / B - V')
ax.set_ylabel('Temperature \ K')
ax.set_title('Colour Feature vs. Temperature')
plt.show()
	#plot the colour feature of each curve against its temperature

'''
change colour map to something more easily seen and read temp dep colour
add colour bar
'''

clf = RandomForestClassifier()
	#load the random forest clssifier

kf = cross_validation.KFold(n = len(colour), n_folds = 5, shuffle = True)
	#use kfolds to split the data

for train_index, test_index in kf:
	#cycle through each kfold and use it as a training set for the algorithm, using the remaining folds as test sets
	X_train, X_test = colour[train_index], colour[test_index]
	y_train, y_test = temp[train_index], temp[test_index]
		#split the data into the given folds (need data in an sp.array for indexing to work)
	clf = clf.fit(X_train, y_train)
		#fit the model the the current training set
	test_pred = clf.predict(X_test)
		#Use the model to predict the temperatures of the test set

	fig, ax = plt.subplots(2,2)
	fig.canvas.set_window_title('Random Forest Temperature Model for Black Body Curves')

	ax[0][0].scatter(y_test, test_pred)
	ax[0][0].set_xlabel('Actual temperature \ K')
	ax[0][0].set_ylabel('Predicted temperature \ K')
	ax[0][0].set_title('Actual vs. Predicted temperature')
        	#plot the actual vs. predicted temperature

	error = test_pred - y_test
        	#calculate the error of the fit

	MAD = sp.median([abs(i) for i in(test_pred - sp.median(test_pred))])
		#calculate the MAD of the data

	sns.kdeplot(error, ax=ax[0][1], shade=True)
	ax[0][1].set_title('Error of Prediction')
	ax[0][1].set_xlabel('Absolute Error')
	ax[0][1].set_ylabel('Number')
	ax[0][1].set_title('Absolute Error on Temperature Prediction')
        	#plot the univariant kernel density estimator

	sns.residplot(y_test, test_pred, lowess = True, ax = ax[1][0])
	ax[1][0].set_title('Residuals of Prediction')
	ax[1][0].set_xlabel('Actual Temperature \ K')
	ax[1][0].set_ylabel('Predicted Temperature Residual \ K')
	ax[1][0].set_title('Actual vs. Prediction Residual Temperature')
		#plot the residuals of the predicted temperatures

	ax[1][0].text(sp.amax(y_test)*1.7,0,'MAD = ' + str(MAD))

	ax[1][1].axis('off')

	plt.tight_layout()

plt.show()

'''

fig, ax = plt.subplots()
ax.scatter(df.colour, df.teff)
ax.set_title('Star colour vs. temperature')
ax.set_xlabel('Star colour / log(B/V)')
ax.set_ylabel('Temperature / K')

#plt.savefig('cvt')

#plot colour vs. temp

colour = sp.reshape(df.colour, (-1, 1))
temperature = []

for i in range(len(df.teff)):
    temperature.append(int(df.teff[i]))
        #the temperatures need to be integers

temperature= sp.array(temperature)

clf = RandomForestClassifier()
kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)

accuracy = []

for j in range(1,21):

	accuracy_sum = []
	kf = cross_validation.KFold(n = len(colour), n_folds = 5, shuffle = True)

	for train_index, test_index in kf:
	
		X_train, X_test = colour[train_index], colour[test_index]
		y_train, y_test = temperature[train_index], temperature[test_index]
		clf = clf.fit(X_train, y_train)
		test_pred = clf.predict(X_test)
		accuracy = sp.concatenate((abs(test_pred - y_test)/(y_test*1.0), accuracy))
		
mean_accuracy = sp.mean(accuracy)
std_accuracy = sp.std(accuracy)

clf.fit(colour, temperature)
pred = clf.predict(colour)

error = pred - temperature

fig, ax = plt.subplots()
ax.scatter(temperature, pred)
ax.set_xlabel('Actual temperature \ K')
ax.set_ylabel('Predicted temperature \ K')
ax.set_title('Actual vs. Predicted temperature')

ax.text(10000, 4000, '{} +/- {}'.format(mean_accuracy, std_accuracy), size = 15, ha = 'right')

#plt.savefig('fit')

#plot actual vs. model temp


sns.residplot(temperature, pred, lowess=True)

#plt.savefig('residue')

for i in range(len(pred)):
	if (abs(pred[i] - temperature[i])/(temperature[i]*1.0)) > 0.1:
		plt.figure()
		plt.plot(df.loc[i].wavelength, df.loc[i].flux, color = 'k')
		plt.axvline(3980)
		plt.axvline(4920)
		plt.axvline(5070)
		plt.axvline(5950)
		#plt.savefig('spectra' + str(i))

plt.show()
'''