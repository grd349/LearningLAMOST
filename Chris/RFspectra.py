import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

clf = RandomForestClassifier()

sfile = 'my_data.csv'
df = pd.read_csv(sfile, sep=',')

colour = np.reshape(df.feature, (-1, 1))

temps = np.array(df["teff"].tolist())

for i in range(len(temps)):
    temps[i] = int(temps[i])

accuracy = []

for j in range(0,21):  
    
    X_train, X_test, y_train, y_test = train_test_split(colour, temps, test_size=0.33, random_state=42)
    clf = clf.fit(X_train,y_train)
    test_pred = clf.predict(X_test)
    acc = abs(test_pred - y_test)/(y_test*1.0)
    accuracy.append(acc)
    """
    for i in range(len(acc)):
        if acc[i] > 0.1:
            plt.plot(df.loc["wavelength"],))
        """    
print X_test   
    
mean_accuracy = np.mean(accuracy)
std_accuracy = np.std(accuracy)

fig, ax = plt.subplots()
ax.scatter(temps, clf.predict(colour))
ax.set_xlabel('Actual Temperature / K')
ax.set_ylabel('Predicted Temperature / K')
ax.set_title('Plot of Predicted vs. Actual Temperature for Spectra')
ax.text(9000, 4000, '{} +/- {}'.format(round(mean_accuracy,3), round(std_accuracy,3)), size = 15, ha = 'right')
plt.savefig('RFspectra')
#plt.show()
