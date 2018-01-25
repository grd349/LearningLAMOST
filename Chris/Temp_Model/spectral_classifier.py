import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from astropy.stats import mad_std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class Spectral_Classifier():
	def __init__(self, sfile):
		''' Reads in dataframe and returns features as a 2D numpy array '''
		df = pd.read_csv(sfile, sep=',')
		self.names = df.columns[1:9]
		self.features = df.as_matrix(self.names)
		self.spec_class = df['CLASS'].values

	def tune_hyperparameters(self, verbose=False):
		''' Uses an exhaustive grid search to tune the hyperparameters of a random forest classifier '''
		parameter_grid = [{'n_estimators':[1,10,20,40,60,80,100],'max_depth':[1,10,100,1000]}]
		clf = GridSearchCV(RandomForestClassifier(), parameter_grid, cv=5)
		clf.fit(self.features, self.spec_class)
		if verbose:
			print(clf.best_params_)
		return clf.best_params_

	def predict_class(self, clf=RandomForestClassifier()):
		''' Fits a classifier to the data and returns model predictions and true classifications of a test set '''
		X_train, X_test, y_train, self.y_test = train_test_split(self.features,self.spec_class,test_size=0.5)
		clf = clf.fit(X_train,y_train)
		self.y_test_pred = clf.predict(X_test)
		print(self.y_test_pred)
		for i in range(len(self.names)):
			if i == 0:
				print("Importances:")
			print(self.names[i] + ' : ', clf.feature_importances_[i])

if __name__ == "__main__":
	sfile = 'test_dataframe'
	spec_clf = Spectral_Classifier(sfile)
	spec_clf.predict_class()

	
"""
#Calculates the median absolute deviation
MAD = mad_std(error)

index = np.argmax(abs(error))
row_index = df.loc[df['designation']==desig_test[index]].index[0]

finalSpectrum = readFits.Spectrum('/data2/cpb405/DR1_3/' + df.get_value(row_index,'filename'))  

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
"""

"""
def plot_results(self, clf='Random Forest Classifier'):
fig, ax = plt.subplots(2,2)
fig.suptitle(clf,y=1.03,fontsize=18)

''' Predicted vs. actual temperature '''
ax[0][0].scatter(true_temp, predicted_temp)
ax[0][0].set_xlabel('Actual Temperature / K')
ax[0][0].set_ylabel('Predicted Temperature / K')
ax[0][0].set_title('Predicted vs. Actual Temperature')
"""