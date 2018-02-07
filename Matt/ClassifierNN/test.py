from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sb


conf = np.loadtxt('Files/confusion.csv', delimiter = ',')

classes = ['STAR', 'GALAXY', 'QSO', 'Unknown']

actual = [[row[i] for row in conf] for i in range(len(classes))]

a = ['Matt', 'Chris', 'Chris', 'Matt', 'Chris']

hot = LabelEncoder()

print(hot.fit_transform(a))

#actual = actual.reshape

ax, fig = plt.subplots()