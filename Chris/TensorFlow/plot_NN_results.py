import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

accbatch = np.genfromtxt('Files/AccBatch1.csv', delimiter=',')
batch = accbatch[:,0]
accuracy = accbatch[:,1]

labs = np.genfromtxt('Files/Labels1.csv', delimiter=',', dtype=str)
preds = np.genfromtxt('Files/Predictions1.csv', delimiter=',')

true_false = []
true = 0
false = 0
for idx in range(len(preds)):
    if preds[idx] == 0:
        true_false.append("Incorrect")
        false+=1
    else:
        true_false.append("Correct")
        true+=1

d = {'Labels':labs,'Predictions':true_false}
df = pd.DataFrame(data=d)

fig, ax = plt.subplots(2,2,figsize=(13,10))
fig.suptitle("Convolutional Neural Net",y=0.96,fontsize=24)

sns.factorplot(x='Labels', hue='Predictions', data=df, size=6, kind='count', palette='muted', ax=ax[0][0])
ax[0][0].set_xlabel("Class")
ax[0][0].set_ylabel("Counts")
ax[0][0].set_title("Predictions for Each Class")

ax[0][1].plot(batch, accuracy)
ax[0][1].set_xlabel("Batch number")
ax[0][1].set_ylabel("Accuracy")
ax[0][1].set_title("Accuracy with Training Batch")
#plt.savefig("AccPlotnoNoise")

ax[1][0].pie([true,false],labels=["Correct","Incorrect"], autopct="%1.2f%%")
ax[1][0].axis("equal")
ax[1][0].set_title("Total Number of Correct/Incorrect Predictions")

plt.tight_layout()
#plt.savefig("MKNoiseResults")
plt.show()