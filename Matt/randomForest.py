import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from run import *

n = 2

patients, headers = get_data()
bmi, ductwidth, popf = clean3(headers, patients)
ductwidth = ductwidth ** n
bmi = bmi ** n
X = np.column_stack([bmi,ductwidth])

min_depth = 1
max_depth = 10
acc = [0]
for i in range(min_depth,max_depth+1):
    clf = RandomForestClassifier(max_depth=i, n_estimators=1, max_features=1)
    temp_accuracy = cross_validation.cross_val_score(clf, X, popf, cv=10)
    if np.mean(acc) < np.mean(temp_accuracy):
        acc = temp_accuracy
        l = i
        
min_estimators = 1
max_estimators = 20
acc = [0]
for i in range(min_estimators,max_estimators+1):
    clf = RandomForestClassifier(max_depth=l, n_estimators=i, max_features=1)
    temp_accuracy = cross_validation.cross_val_score(clf, X, popf, cv=10)
    if np.mean(acc) < np.mean(temp_accuracy):
        acc = temp_accuracy
        m = i

print l
print m

clf = RandomForestClassifier(max_depth=l, n_estimators=m, max_features=1)

total_accuracy = []
total_std = []

for j in range(1,21):
    
    accuracy_sum = []
    kf = cross_validation.KFold(n=408, n_folds=5, shuffle=True)
    
    for train_index, test_index in kf:
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = popf[train_index], popf[test_index]        
        clf = clf.fit(X_train,y_train)
        test_pred = clf.predict(X_test)
        a = accuracy(y_test,test_pred,False)
        accuracy_sum.append(a)
                
    total_accuracy.append(np.mean(accuracy_sum))
    total_std.append(np.std(accuracy_sum))

mean_accuracy = np.mean(total_accuracy)
std_accuracy = np.mean(total_std)

clf = clf.fit(X, popf)

x_min, x_max = 0, 60 ** n
y_min, y_max = 0, 10 ** n
xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100.0),
                     np.arange(y_min, y_max, (y_max-y_min)/100.0))

Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=[8,6])
ax = fig.add_subplot(1,1,1)
v = np.linspace(0, 1.0, 11, endpoint=True)
CS = ax.contourf(xx, yy, Z, v, alpha=0.8, cmap="Reds")
ax.scatter(X[:,0][popf<0.5],X[:, 1][popf<0.5],c='w',label='Success')  
ax.scatter(X[:,0][popf>0.5],X[:, 1][popf>0.5],c='Red',label='Failure')                           
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_title("Naive Bayes Final Model")
ax.set_xlabel("(BMI)^2")
ax.set_ylabel("(Pancreatic Duct Width)^2 / mm^2")
ax.legend()
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel("Probability of Pancreatic Fistula")
ax.text(x_max - 150, y_min + 4, ("%0.2f +/- %0.2f" % 
(mean_accuracy, std_accuracy)), size=15, ha="right")
