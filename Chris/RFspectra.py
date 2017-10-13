import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
                        
clf = RandomForestClassifier()

sfile = 'my_data.csv'

df = pd.read_csv(sfile, sep=',')

colour = np.reshape(df.feature, (-1, 1))
temps = df.teff

for i in range(len(temps)):
    temps[i] = int(temps[i])

kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)

error = 0

for train_index, test_index in kf:   
    X_train, X_test = colour[train_index], colour[test_index]
    y_train, y_test = temps[train_index], temps[test_index]
    clf = clf.fit(X_train,y_train)
    test_pred = clf.predict(X_test)
    
    """
    error = test_pred - y_test
    fig, ax2 = plt.subplots()
    ax2.scatter(y_test, test_pred)
    ax2.set_xlabel('Actual Temperature / K')
    ax2.set_ylabel('Predicted Temperature / K')
    ax2.set_title('Plot of Predicted vs. Actual Temperature for Modelled Blackbodies')
    plt.show()
    
    fig, ax3 = plt.subplots()
    ax3.hist(error)
    ax3.set_xlabel('Actual Temperature / K')
    ax3.set_ylabel('Predicted Temperature / K')
    ax3.set_title('Plot of Predicted vs. Actual Temperature for Modelled Blackbodies')
    plt.show()
    """

clf.fit(colour, temps)
pred = clf.predict(colour)

error = pred - temps

fig, ax2 = plt.subplots()
ax2.scatter(temps, clf.predict(colour))
ax2.set_xlabel('Actual Temperature / K')
ax2.set_ylabel('Predicted Temperature / K')
ax2.set_title('Plot of Predicted vs. Actual Temperature for Spectra')
plt.show()

fig, ax3 = plt.subplots()
ax3.hist(error)
ax3.set_xlabel('Actual Temperature / K')
ax3.set_ylabel('Predicted Temperature / K')
ax3.set_title('Plot of Predicted vs. Actual Temperature for Spectra')
plt.show()
