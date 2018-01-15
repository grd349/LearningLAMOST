import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob

from astropy.stats import mad_std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from read_fits import Spectrum

class Temperature_Regressor():
    def __init__(self, direc):
        ''' Reads in dataframe and returns features as a 2D numpy array '''
        features_df = pd.concat([pd.read_csv(f, sep=',') for f in glob.glob(direc + '/*.csv')])
        
        catalog_file = "/data2/cpb405/dr1_stellar.csv"
        catalog = pd.read_csv(catalog_file, sep='|')
        catalog.drop_duplicates(subset = 'designation', inplace = True)
        self.df = catalog.merge(features_df, on='designation', how='inner')        

    def tune_hyperparameters(self, verbose=False):
        ''' Uses an exhaustive grid search to tune the hyperparameters of a random forest regressor '''
        parameter_grid = [{'n_estimators':[1,10,20,40,60,80,100],'max_depth':[1,10,100,1000]}]
        regr = GridSearchCV(RandomForestRegressor(), parameter_grid, cv=5)
        regr.fit(self.features, self.temperatures)
        if verbose:
            print(regr.best_params_)
        return regr.best_params_
    
    def extract_features(self, df):
        self.names = df.columns[38:50]
        features = df.as_matrix(columns = self.names)
        self.names = [name[1:] for name in self.names]
        temp_features = features[:,1:5]
        temp_names = self.names[1:5]
        for idx in range(len(temp_features.T)-1):
            for j in np.arange(idx+1, len(temp_features.T)):
                d = np.reshape(temp_features[:,idx] - temp_features[:,j],(-1, 1))
                features = np.hstack((features,d))
            self.names = np.append(self.names,[temp_names[idx] + "-" + i for i in temp_names[idx+1:]])
        return features
                                 
    def predict_temperatures(self, regr=RandomForestRegressor(n_estimators=80, max_depth=10, max_features=8)):
        ''' Fits a regressor to the data and returns model predictions and true values of a test set '''
        train_df, self.test_df = train_test_split(self.df,test_size=0.5)      
        X_train = self.extract_features(train_df)
        X_test = self.extract_features(self.test_df)
        y_train = train_df['teff'].values
        self.y_test = self.test_df['teff'].values
        regr = regr.fit(X_train,y_train)
        self.y_test_pred = regr.predict(X_test)
        self.error = self.y_test_pred - self.y_test
        self.importances = regr.feature_importances_
        for i in range(len(self.names)):
            print("Importances:")
            print(self.names[i] + ' : ', self.importances[i])
    
    def blackbody(self, T, wavelength, counts):
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        E = (8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))
        normalise = counts/np.sum(E)
        return normalise*E

    def plot_results(self, regr='Random Forest Regressor'):
        fig, ax = plt.subplots(2,2)
        fig.suptitle(regr,y=1.03,fontsize=18)
        
        ends = [np.amin(self.y_test), np.amax(self.y_test)]
        ''' Predicted vs. actual temperature '''
        ax[0][0].scatter(self.y_test, self.y_test_pred)
        ax[0][0].plot(ends, ends, ls = ':', color = 'red')
        ax[0][0].set_xlabel('Actual Temperature / K')
        ax[0][0].set_ylabel('Predicted Temperature / K')
        ax[0][0].set_title('Predicted vs. Actual Temperature')
        ax[0][0].annotate('MAD = {0:.2f}'.format(mad_std(self.error)), xy=(0.05, 0.90), xycoords='axes fraction', color='r')
        
        ''' Kernel density estimator '''
        sns.kdeplot(self.error, ax=ax[0][1], shade=True)
        ax[0][1].set_xlabel('Absolute Error / K')
        ax[0][1].set_ylabel('Fraction of Points with Error')
        ax[0][1].set_title('KDE Plot for Absolute Errors')
        
        ''' Importances bar plot '''
        sns.barplot(self.names, self.importances, ax=ax[1][0])
        ax[1][0].set_xlabel('Feature')
        ax[1][0].set_ylabel('Importance')
        ax[1][0].set_title('Importances of Spectral Features')
        for tick in ax[1][0].get_xticklabels():
            tick.set_rotation(90)
        
        outlier_index = np.argmax(abs(self.error))
        desig_test = self.test_df['designation'].values
        outlier_row = self.test_df.loc[self.test_df['designation']==desig_test[outlier_index]].index[0]
        outlier = Spectrum('/data2/mrs493/DR1_3/' + self.test_df.get_value(outlier_row,'FILENAME'))
        
        blackbody_true = self.blackbody(self.test_df.get_value(outlier_row,'teff'), outlier.wavelength, np.nansum(outlier.flux))
        blackbody_pred = self.blackbody(self.y_test_pred[outlier_index], outlier.wavelength, np.nansum(outlier.flux))
        
        ''' Outlier spectrum plot '''
        ax[1][1].plot(outlier.wavelength, outlier.flux)
        ax[1][1].plot(outlier.wavelength,blackbody_true,'--',c='r',label='True Temp.')
        ax[1][1].plot(outlier.wavelength,blackbody_pred,'--',c='b',label='Predicted Temp.')
        ax[1][1].set_xlabel('Wavelength / Angstroms')
        ax[1][1].set_ylabel('Flux')
        ax[1][1].set_title('Spectrum for Greatest Outlier')
        ax[1][1].legend()
                
        plt.tight_layout()
        plt.show()
        	
if __name__ == "__main__":
    direc = 'TempCSVs1'
    spec_regr = Temperature_Regressor(direc)
    spec_regr.predict_temperatures()
    spec_regr.plot_results()