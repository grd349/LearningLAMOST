from astropy.io import fits
import scipy as sp
import matplotlib.pyplot as plt
import glob
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

#from fits import Spectra

def blackbody(T):
    wavelength = sp.linspace(3000, 9000, 3001)
    h = 6.63e-34
    c = 3e8
    k = 1.38e-23
    return (8*sp.pi*h*c)/((wavelength*1e-10)**5*(sp.exp(h*c/((wavelength*1e-10)*k*T))-1))
	#calculate a blackboduy curve for a given temperature

temp = sp.array(sp.random.normal(6000, 2000, 800))
	#generate a normal distribution of tempratures

for i in range(len(temp)):
    temp[i] = int(int(temp[i]))
	#convert the temperatures to (positive) integers for use in the algorithm

colour = []

for t in temp:
    spectra = blackbody(t)
    B = -2.5*sp.log10(sp.sum(spectra[490:960]))
    V = -2.5*sp.log10(sp.sum(spectra[1035:1475]))
    colour.append(B - V)
	#for each temperature, generate a balckbody curve then calculate its colour features

colour = sp.reshape(colour, (-1, 1))
	#reshape the colour to a column vector for use in the algorithm


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
