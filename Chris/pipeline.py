import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.pipeline import make_pipeline

#Reads in dataframe
sfile = 'spectra_dataframe.csv'
df = pd.read_csv(sfile, sep=',')

#Reads in temperature and features from dataframe
BminusV = np.array(df["BminusV"].tolist())
BminusR = np.array(df["BminusR"].tolist())
VminusR = np.array(df["VminusR"].tolist())
totCounts = np.array(df["totCounts"].tolist())

features = np.column_stack((BminusV,BminusR,VminusR,totCounts))

temps = np.array(df["teff"].tolist())
desig = np.array(df["designation"].tolist())
    
features_train, features_test, temp_train, temp_test = train_test_split(features, temps, test_size=0.5)


rf = RandomForestRegressor()
ada = AdaBoostRegressor()

pipeline = make_pipeline(rf,ada)
pipeline.fit(features_train,temp_train)

fig, ax = plt.subplots()
ax.scatter(temp_test, pipeline.predict(features_test))
ax.set_xlabel('Actual Temperature / K')
ax.set_ylabel('Predicted Temperature / K')
ax.set_title('Predicted vs. Actual Temp')


print(pipeline.predict(features_test))