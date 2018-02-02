import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    conf = np.loadtxt('Files/confusion.csv', delimiter = ',')
    
    classes = ['STAR', 'GALAXY', 'QSO', 'Unknown']

fig, ax = plt.subplots()

colours = ['y', 'm', 'r', 'g']

for i in range(len(classes) - 1):
    conf[i+1] = conf[i+1] + conf[i]*1.1

for i in range(len(classes))[::-1]:
    sns.barplot(classes, conf[i], ax = ax, color = colours[i], label = classes[i])

ax.set_title('Neural Network Classification of Stellar Spectrum')
ax.set_xlabel('LAMOST Classification')
ax.set_ylabel('Model Classification')
ax.legend()
plt.savefig('NNclassifications.pdf')
plt.show()
