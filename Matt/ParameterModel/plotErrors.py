from fits import Spectrum
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

spectra = pd.read_csv('errors.csv')

for spectrum in spectra['file'].tolist():
    spec = Spectrum('/data2/mrs493/DR1_3/' + spectrum)
    fig, ax = plt.subplots()
    spec.plotFlux(ax = ax)
    ax.set_xlabel('Wavelength \ Angstrom')
    ax.set_ylabel('Flux')
    ax.set_title(spectrum[:-5])
    plt.show()
