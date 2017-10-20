import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

clf = RandomForestClassifier()

#Reads in dataframe
sfile = 'spectra_dataframe.csv'
df = pd.read_csv(sfile, sep=',')

#Reads in temperature and B-V from dataframe
colour = np.reshape(df.feature, (-1, 1))

temps = np.array(df["teff"].tolist())

for i in range(len(temps)):
    temps[i] = int(temps[i])

accuracy = []

#Runs random forest algorithm 20 times using simple train_test_split cross-validation method
for j in range(0,20):  
    
    X_train, X_test, y_train, y_test = train_test_split(colour, temps, test_size=0.33, random_state=42)
    clf = clf.fit(X_train,y_train)
    test_pred = clf.predict(X_test)
    
    #Calculates accuracy of each model and adds to list
    acc = abs(test_pred - y_test)/(y_test*1.0)
    accuracy.append(acc)

#Calulcates the mean and standard deviation from accuracy list
mean_accuracy = np.mean(accuracy)
std_accuracy = np.std(accuracy)

clf.fit(colour, temps)
pred = clf.predict(colour)

error = pred - temps

#Predicts temperature for all points and plots this against true temperatures
fig, ax1 = plt.subplots()
ax1.scatter(temps, pred)
ax1.set_xlabel('Actual Temperature / K')
ax1.set_ylabel('Predicted Temperature / K')
ax1.set_title('Prediction of Spectral Temperature using Random Forest Classifier')
ax1.text(9000, 4000, '{} +/- {}'.format(round(mean_accuracy,3), round(std_accuracy,3)), size = 15, ha = 'right')
plt.savefig('RFspectraOverfit')
#plt.show()

fig, ax2 = plt.subplots()
sns.kdeplot(error, ax=ax2, shade=True)
ax2.set_xlabel('Absolute Error / K')
ax2.set_ylabel('Fraction of Points with Given Error')
ax2.set_title('Kernel Density Estimator for Absolute Errors on Random Forest Model')
plt.savefig('RFspectraOverfitkde')
#plt.show()

fig, ax3 = plt.subplots()
sns.residplot(temps, pred, lowess=True, ax=ax3)
ax3.set_xlabel('Actual Temperature / K')
ax3.set_ylabel('Residual of Fit')
ax3.set_title('Residual Plot for Predicted vs. Actual Spectral Temperature')
plt.savefig('RFspectraOverfitresid')
#plt.show()

