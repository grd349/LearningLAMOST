from astropy.io import fits
import scipy as sp
import matplotlib.pyplot as plt
import glob
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

from fits import Spectra

def blackbody(T):
    wavelength = sp.linspace(3000, 9000, 3001)
    h = 6.63e-34
    c = 3e8
    k = 1.38e-23
    return (8*sp.pi*h*c)/((wavelength*1e-10)**5*(sp.exp(h*c/((wavelength*1e-10)*k*T))-1))

temp = sp.array(sp.random.normal(6000, 2000, 800))

colour = []

for t in temp:
    spectra = blackbody(t)
    B = -2.5*sp.log10(sp.sum(spectra[490:960]))
    V = -2.5*sp.log10(sp.sum(spectra[1035:1475]))
    colour.append(B - V)

colour = sp.reshape(colour, (-1, 1))

for i in range(len(temp)):
    temp[i] = int(temp[i])

fig, ax = plt.subplots()
ax.scatter(colour, temp)
ax.set_xlabel('Colour Feature / B - V')
ax.set_ylabel('Temperature \ K')
ax.set_title('Colour Feature vs. Temperature')
plt.show()

clf = RandomForestClassifier()

kf = cross_validation.KFold(n = len(colour), n_folds = 5, shuffle = True)

for train_index, test_index in kf:

	X_train, X_test = colour[train_index], colour[test_index]
	y_train, y_test = temp[train_index], temp[test_index]
	clf = clf.fit(X_train, y_train)
	test_pred = clf.predict(X_test)

	fig, ax = plt.subplots(2,2)
	fig.canvas.set_window_title('Random Forest Temperature Model for Black Body Curves')

	ax[0][0].scatter(y_test, test_pred)
	ax[0][0].set_xlabel('Actual temperature \ K')
	ax[0][0].set_ylabel('Predicted temperature \ K')
	ax[0][0].set_title('Actual vs. Predicted temperature')

	error = test_pred - y_test

	sns.kdeplot(error, ax=ax[0][1], shade=True)
	ax[0][1].set_title('Error of Prediction')
	ax[0][1].set_xlabel('Absolute Error')
	ax[0][1].set_ylabel('Number')
	ax[0][1].set_title('Absolute Error on Temperature Prediction')

	sns.residplot(y_test, test_pred, lowess = True, ax = ax[1][0])
	ax[1][0].set_title('Residuals of Prediction')
	ax[1][0].set_xlabel('Actual Temperature \ K')
	ax[1][0].set_ylabel('Predicted Temperature Residual \ K')
	ax[1][0].set_title('Actual vs. Prediction Residual Temperature')

	plt.tight_layout()

plt.show()
'''
clf.fit(colour, temp)

pred = clf.predict(colour)

error = pred - temp

fig, ax = plt.subplots()
ax.scatter(temp, pred)
ax.set_xlabel('Actual temperature \ K')
ax.set_ylabel('Predicted temperature \ K')
ax.set_title('Actual vs. Predicted temperature')

fig, ax = plt.subplots()
sns.kdeplot(error, ax=ax, shade=True)
ax.set_title('Error of Prediction')
plt.show()
'''
