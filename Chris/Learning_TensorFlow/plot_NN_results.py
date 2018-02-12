import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

data1 = np.genfromtxt('AccBatch1.csv', delimiter=',')
batch = data1[:,0]
accuracy = data1[:,1]

fig, ax = plt.subplots()
ax.plot(batch, accuracy)
ax.set_xlabel("Batch number")
ax.set_ylabel("Accuracy")
plt.savefig("AccPlotnoNoise")
"""
sns.set(style="whitegrid")

data2 = np.genfromtxt('PredLabs.csv', delimiter=',')

# Draw a nested barplot to show survival for class and sex
g = sns.factorplot(x="class", y="survived", hue="sex", data=titanic,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
"""