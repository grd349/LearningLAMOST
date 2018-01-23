import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob

from astropy.stats import mad_std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from read_fits import Spectrum

class Temperature_Regressor():
    def __init__(self, direc, bright = False):
        ''' Reads in dataframe and returns features as a 2D numpy array '''
        features_df = pd.concat([pd.read_csv(f, sep=',') for f in glob.glob(direc + '/*.csv')])       
        catalog_file = "/data2/cpb405/dr1_stellar.csv"
        catalog = pd.read_csv(catalog_file, sep='|')
        catalog.drop_duplicates(subset = 'designation', inplace = True)
        self.df = catalog.merge(features_df, on='designation', how='inner')
        if bright:
            self.df = self.df.sort_values('cAll')[:1000]
        self.names = self.df.columns[38:55]
        imputer = Imputer(missing_values = 0)
        self.df[self.names] = imputer.fit_transform(self.df[self.names])

    def tune_hyperparameters(self, feats, temps, verbose=False):
        ''' Uses an exhaustive grid search to tune the hyperparameters of a random forest regressor '''
        parameter_grid = [{'n_estimators':[20,40,60,80,100],'max_depth':[1000,2000,3000,4000,5000],'max_features':[6,9,12,15]}]
        regr = GridSearchCV(RandomForestRegressor(), parameter_grid, cv=5)
        regr.fit(feats, temps)
        if verbose:
            print(regr.best_params_)
        return regr.best_params_
    
    def extract_features(self):
        ''' Carries out feature engineering on the photometric and equivalent width features '''
        features = self.df.as_matrix(columns = self.names)         
        col_features = features[:,1:5]
        col_names = self.names[1:5]
        for idx in range(len(col_features.T)-1):
            for j in np.arange(idx+1, len(col_features.T)):
                d = np.reshape(col_features[:,idx] - col_features[:,j],(-1, 1))
                n = col_names[idx][1:] + "-" + col_names[j][1:]
                self.df[n] = d
            self.names = np.append(self.names,[col_names[idx][1:] + "-" + i[1:] for i in col_names[idx+1:]])                     
        lin_features = features[:,5:14]
        lin_names = self.names[5:14]
        for idx in range(len(lin_features.T)-1):
            for j in np.arange(idx+1, len(lin_features.T)):
                d = np.reshape(lin_features[:,idx] / lin_features[:,j],(-1,1))
                n = lin_names[idx][1:] + "/" + lin_names[j][1:]
                self.df[n] = d
            self.names = np.append(self.names,[lin_names[idx][1:] + "/" + i[1:] for i in lin_names[idx+1:]])
        scaler = StandardScaler()
        self.df[self.names] = scaler.fit_transform(self.df[self.names])
                                 
    def predict_temperatures(self, model, tune=False, verbose=False):
        ''' Fits a regressor to the data and returns model predictions and true values of a test set '''
        train_df, self.test_df = train_test_split(self.df,test_size=0.5)
        X_train = train_df.as_matrix(columns = self.names)
        X_test = self.test_df.as_matrix(columns = self.names)
        y_train = train_df[model].values
        self.y_test = self.test_df[model].values 
        if tune:
            hyp = self.tune_hyperparameters(X_train,y_train,True)
            self.max_features = hyp['max_features']
            regr = RandomForestRegressor(n_estimators = hyp['n_estimators'], max_depth = hyp['max_depth'], max_features = self.max_features)   
        else:
            self.max_features = 15
            regr = RandomForestRegressor(n_estimators=100, max_depth=3000, max_features=self.max_features)     
        regr = regr.fit(X_train,y_train)        
        self.y_test_pred = regr.predict(X_test)
        self.error = self.y_test_pred - self.y_test
        self.importances = regr.feature_importances_
        if verbose:
            for i in range(len(self.names)):
                print("Importances:")
                print(self.names[i] + ' : ', self.importances[i])
    
    def blackbody(self, T, wavelength, counts):
        ''' Models an ideal blackbody curve of a given temperature '''
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        E = (8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))
        normalise = counts/np.sum(E)
        return normalise*E

    def plot_results(self, model, regr='Random Forest Regressor'):
        ''' Creates a 2 by 2 grid of subplots presenting the predictions of the RFR '''
        fig, ax = plt.subplots(2,2)
        fig.suptitle(regr + '\n' + model,y=1.03,fontsize=18)
        
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
        
        names = self.names.copy()
        for i in range(len(names)):
            if names[i][0] == 'c' or names[i][0] == 'l':
                names[i] = names[i][1:]
                
        imp = [[i,n] for i,n in sorted(zip(self.importances, names), reverse = True)]        
        ''' Importances bar plot '''
        sns.barplot([i[1] for i in imp][:self.max_features], [i[0] for i in imp][:self.max_features], ax = ax[1][0])
        ax[1][0].set_xlabel('Feature')
        ax[1][0].set_ylabel('Importance')
        ax[1][0].set_title('Importances of Spectral Features')
        for tick in ax[1][0].get_xticklabels():
            tick.set_rotation(90)
        
        outlier_index = np.argmax(abs(self.error))
        desig_test = self.test_df['designation'].values
        outlier_row = self.test_df.loc[self.test_df['designation']==desig_test[outlier_index]].index[0]
        outlier = Spectrum('/data2/mrs493/DR1_3/' + self.test_df.get_value(outlier_row,'FILENAME'))      
        
        ''' Outlier spectrum plot '''
        ax[1][1].plot(outlier.wavelength, outlier.flux)
        if model == 'teff':
            blackbody_true = self.blackbody(self.test_df.get_value(outlier_row,'teff'), outlier.wavelength, np.nansum(outlier.flux))
            blackbody_pred = self.blackbody(self.y_test_pred[outlier_index], outlier.wavelength, np.nansum(outlier.flux))
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
    spec_regr = Temperature_Regressor(direc, bright = True)
    spec_regr.extract_features()
    models = ['teff', 'logg', 'feh']
    for mod in models:
        spec_regr.predict_temperatures(tune = False, model = mod)
        spec_regr.plot_results(model = mod)