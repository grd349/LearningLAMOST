import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob

from astropy.stats import mad_std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

class Temperature_Regressor():
    def __init__(self, direc):
        ''' Reads in dataframe and returns features as a 2D numpy array '''
        features_df = pd.concat([pd.read_csv(f, sep=',') for f in glob.glob(direc + '/*.csv')])
        
        catalog_file = "/data2/cpb405/dr1_stellar.csv"
        catalog = pd.read_csv(catalog_file, sep='|')
        catalog.drop_duplicates(subset = 'designation', inplace = True)
        self.df = catalog.merge(features_df, on='designation', how='inner')
        
        self.names = self.df.columns[38:46]
        self.features = self.df.as_matrix(columns = self.names)
        self.designations = self.df['designation'].values
        self.temperatures = self.df['teff'].values

    def tune_hyperparameters(self, verbose=False):
        ''' Uses an exhaustive grid search to tune the hyperparameters of a random forest regressor '''
        parameter_grid = [{'n_estimators':[1,10,20,40,60,80,100],'max_depth':[1,10,100,1000]}]
        regr = GridSearchCV(RandomForestRegressor(), parameter_grid, cv=5)
        regr.fit(self.features, self.temperatures)
        if verbose:
            print(regr.best_params_)
        return regr.best_params_
    
    def extract_features(self, df):
        features = []
        for feat in df.columns:
            if feat[0] == 'c' or feat[0] == 'l':
                print(feat)
                features.append(df[feat].values)
        print(features)
        return features
                                 
    def predict_temperatures(self, regr=RandomForestRegressor()):
        ''' Fits a regressor to the data and returns model predictions and true values of a test set '''
        train, test = train_test_split(self.df,test_size=0.5)
        X_train = train.as_matrix(columns = self.names)
        X_test = test.as_matrix(columns = self.names)
        y_train = train['teff'].values
        self.y_test = test['teff'].values
        regr = regr.fit(X_train,y_train)
        self.y_test_pred = regr.predict(X_test)
        self.error = self.y_test_pred - self.y_test
        for i in range(len(self.names)):
            if i == 0:
                print("Importances:")
                print(self.names[i] + ' : ', regr.feature_importances_[i])

    def plot_results(self, regr='Random Forest Regressor'):
        fig, ax = plt.subplots(2,2)
        fig.suptitle(regr,y=1.03,fontsize=18)
        
        ''' Predicted vs. actual temperature '''
        ax[0][0].scatter(self.y_test, self.y_test_pred)
        ax[0][0].set_xlabel('Actual Temperature / K')
        ax[0][0].set_ylabel('Predicted Temperature / K')
        ax[0][0].set_title('Predicted vs. Actual Temperature')
        
        ''' Kernel density estimator '''
        sns.kdeplot(self.error, ax=ax[0][1], shade=True)
        ax[0][1].set_xlabel('Absolute Error / K')
        ax[0][1].set_ylabel('Fraction of Points with Error')
        ax[0][1].set_title('KDE Plot for Absolute Errors')
        
        ''' Residuals of errors '''
        sns.residplot(self.y_test, self.y_test_pred, lowess=True, ax=ax[1][0], line_kws={'color':'red'})
        ax[1][0].set_xlabel('Actual Temperature / K')
        ax[1][0].set_ylabel('Residual of Fit')
        ax[1][0].set_title('Residuals of Errors')
        ax[1][0].annotate('MAD = {0:.2f}'.format(mad_std(self.error)), xy=(0.05, 0.90), xycoords='axes fraction', color='r')
        
                
        plt.tight_layout()
        plt.show()
        	
if __name__ == "__main__":
    direc = 'Output'
    spec_regr = Temperature_Regressor(direc)
    spec_regr.predict_temperatures()
    spec_regr.plot_results()
	
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