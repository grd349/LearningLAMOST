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

features_cont = np.column_stack((BV,BR,BI,VR,VI,RI))
features_line = np.column_stack((HalphaEW,HbetaEW,HgammaEW))

kf = cross_validation.KFold(n=len(BV), n_folds=5, shuffle=True)
j = 1
foldAverage = []

feat_train_cont,feat_test_cont,feat_train_line,feat_test_line,temp_train,temp_test,desig_train,desig_test = train_test_split(features_cont,features_line,temps,desig,test_size=0.5)

#tuned_params = [{'n_estimators':[1,10,20,40,60,80,100],'max_depth':[1,10,100,1000]}]

#clf = GridSearchCV(RandomForestRegressor(), tuned_params, cv=5)
#clf.fit(features_train, temp_train)
#print(clf.best_params_)

clf_cont = RandomForestRegressor(n_estimators=80,max_depth=10)
clf_line = RandomForestRegressor(n_estimators=80,max_depth=10)

#Fits the random forest to the training set and then predicts the temperature of the test set
clf_cont = clf_cont.fit(feat_train_cont,temp_train)
clf_line = clf_line.fit(feat_train_line,temp_train)

test_pred_cont = clf_cont.predict(feat_test_cont)
test_pred_line = clf_line.predict(feat_test_line)

importances_cont = clf_cont.feature_importances_
importances_line = clf_line.feature_importances_
print(importances_cont)
print(importances_line)

fig, ax = plt.subplots()

ax.scatter(test_pred_line,test_pred_cont)
ax.set_xlabel("Temperature Predicted from Spectral Lines")
ax.set_ylabel("Temperature Predicted from Continuum Features")
ax.set_title("Comparison of Temperatures Predicted by Different Feature Types")

for i in range(len(test_pred_cont)):
    if abs(test_pred_cont[i] - test_pred_line[i]) > 1500:
        row_index = df.loc[df['designation']==desig[i]].index[0]
        Spectrum = readFits.Spectrum('/data2/mrs493/DR1/' + df.get_value(row_index,'filename'))
        
        fig, ax = plt.subplots()
        ax.plot(Spectrum.wavelength, Spectrum.flux)
        ax.set_xlabel('Wavelength / Angstroms')
        ax.set_ylabel('Flux')
        ax.set_title('Outlier Spectrum')
        ax.annotate('Type = {}'.format(Spectrum.CLASS), xy=(0.55, 0.08), xycoords='axes fraction',color='r')
