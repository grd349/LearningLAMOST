import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import seaborn as sns

clf = RandomForestClassifier()

#Defines a function which creates a blackbody spectrum using a given temperature
def blackbody(T):
    #Calculates the flux and wavelength for the blackbody
    h = 6.63e-34
    c = 3e8
    k = 1.38e-23
    wavelength = np.linspace(3000,9000,4000)
    E = 1e-4*(8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))
    
    #Calculates the B-V feature for the blackbody
    letters = {"B":[3980,4920], "V":[5070,5950]}
    bandCounts = {"B":0, "V":0}
        
    for letter in letters:
        lower = np.searchsorted(wavelength,letters[letter][0],side="left")
        upper = np.searchsorted(wavelength,letters[letter][1],side="right")       
        bandCounts[letter] = np.sum(E[lower:upper])
    
    return (-2.5 * np.log10(bandCounts["B"]))-(-2.5 * np.log10(bandCounts["V"]))
   sns.kdeplot(error, ax=ax3, shade=True)
    ax3.set_xlabel('Absolute Error / K')
    ax3.set_ylabel('Fraction of Points with Given Error')
    ax3.set_title('Kernel Density Estimator for Absolute Errors on Random Forest Model')
    plt.show()
#Creates an array of 800 random temperatures with mean 6000K, standard deviation 2000K
temps = np.random.normal(6000,2000,800)
temps = np.array([ abs(x) for x in temps])
for i in range(len(temps)):
    temps[i] = int(temps[i])

blackbodyList = []

#Creates an array of blackbody B-V values corresponding to the values in the temperature array
for i in temps:
    blackbodyList.append(blackbody(i))

#Plots temperature against B-V
fig, ax1 = plt.subplots()
ax1.scatter(blackbodyList,temps)
ax1.set_xlabel('B-V Feature')
ax1.set_ylabel('Temperature / K')
ax1.set_title('Plot of Temp vs. B-V Feature for Modelled Blackbody Spectra')
plt.show()
print(np.sum(blackbodyList)) 

colour = np.reshape(blackbodyList, (-1, 1))

kf = cross_validation.KFold(n=len(colour), n_folds=5, shuffle=True)

error = 0

#Uses k-folds to split the data into 5 sets and performs training on 4/5 sets then testing on the 5th set
#in all five ways
for train_index, test_index in kf:   
    X_train, X_test = colour[train_index], colour[test_index]
    y_train, y_test = temps[train_index], temps[test_index]
    
    #Fits the random forest to the training set and then predicts the temperature of the test set
    clf = clf.fit(X_train,y_train)
    test_pred = clf.predict(X_test)
    
    #Calculates absolute error on each point
    error = test_pred - y_test
    
    #Plots machine learned temperature against actual temperature of the spectra
    fig, ax2 = plt.subplots()
    ax2.scatter(y_test, test_pred)
    ax2.set_xlabel('Actual Temperature / K')
    ax2.set_ylabel('Predicted Temperature / K')
    ax2.set_title('Plot of Predicted vs. Actual Temperature for Modelled Blackbodies')
    plt.show()
    
    #Plots a kernel density estimator plot for the absolute errors
    fig, ax3 = plt.subplots()
    sns.kdeplot(error, ax=ax3, shade=True)
    ax3.set_xlabel('Absolute Error / K')
    ax3.set_ylabel('Fraction of Points with Given Error')
    ax3.set_title('Kernel Density Estimator for Absolute Errors on Random Forest Model')
    plt.show()
    

"""
clf.fit(colour, temps)
pred = clf.predict(colour)

error = pred - temps

fig, ax2 = plt.subplots()
ax2.scatter(temps, clf.predict(colour))
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
