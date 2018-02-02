#!/usr/bin/env python3

from fits import Spectrum

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

sfile = '/data2/mrs493/my_data2.csv'
df = pd.read_csv(sfile, sep=',')


KNR = neighbors.KNeighborsRegressor()
RNR = neighbors.RadiusNeighborsRegressor()
RFR = RandomForestRegressor()
GNB = GaussianNB()

pipeline = make_pipeline(RFR)
train, test = train_test_split(df, test_size=0.2)


colour_train = sp.reshape(train.colour.tolist(), (-1, 1))
colour_test = sp.reshape(test.colour.tolist(), (-1, 1))

temp_train = train.teff.tolist()
temp_test = test.teff.tolist()


pipeline.fit(colour_train, temp_train)
    #fit the model the the current training set
    
temp_pred = pipeline.predict(colour_test)
    #Use the model to predict the temperatures of the test set

fig, ax = plt.subplots(2,2)

fig.suptitle('')
    
ax[0][0].scatter(temp_test, temp_pred)
ax[0][0].set_xlabel('Actual temperature \ K')
ax[0][0].set_ylabel('Predicted temperature \ K')
ax[0][0].set_title('Actual vs. Predicted temperature')
    #plot the actual vs. predicted temperature

error = temp_pred - temp_test
    #calculate the error of the fit

MAD = stats.mad_std(error)
    #calculate the MAD of the data

sns.kdeplot(error, ax=ax[0][1], shade=True)
ax[0][1].set_xlabel('Absolute Error')
ax[0][1].set_ylabel('Fraction of Points with\nGiven Error')
ax[0][1].set_title('KDE of Absolute Error\non Temperature Prediction')
    #plot the univariant kernel density estimatorplt.axvline(letters[letter][0])
    
sns.residplot(sp.array(temp_test), sp.array(temp_pred), lowess = True, ax = ax[1][0], line_kws={'color': 'red'})
ax[1][0].set_title('Residuals of Prediction')
ax[1][0].set_xlabel('Actual Temperature \ K')
ax[1][0].set_ylabel('Prediction Residual \ K')
    #plot the residuals of the predicted temperatures
ax[1][0].annotate('MAD = {0:.2f}'.format(MAD), xy = (0.05, 0.90), xycoords = 'axes fraction')
    #write the MAD in the plot

'''
look into residplot - appears residual not propotrional to error (see SGD plot)
'''

test_index = sp.argmax(abs(error))

spectrum = Spectrum('/data2/mrs493/DR1/' + test.filename.tolist()[test_index])
spectrum.plotFlux(ax = ax[1][1], Tpred = temp_pred[test_index], Teff = temp_test[test_index])

ax[1][1].set_xlabel('Wavelength \ Angstroms')
ax[1][1].set_ylabel('Flux')
ax[1][1].set_title('Spectra and model blackbody curve\nfor greatest outlier')
    
plt.tight_layout()

plt.show()

#print sp.mean(importance, 0)

'''
add more classifiers
optimise hyperparameters
add other features i.e. change band widths, move position, use single band etc.
find feature importance
compare cross-validatted to non-cross-validated models
'''

    

