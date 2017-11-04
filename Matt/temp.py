from fits import Spectrum

import time

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy import stats
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

sfile = '/data2/mrs493/my_data2.csv'

df = pd.read_csv(sfile, sep=',')

for i in df.filename:
    spectrum = Spectrum('/data2/mrs493/DR1/' + i)
    if spectrum.CLASS == 'STAR':
        spectrum.plotFlux(titles = True)
        plt.show()
        