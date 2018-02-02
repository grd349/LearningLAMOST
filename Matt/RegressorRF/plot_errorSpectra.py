from fits import Spectrum
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

spectra = pd.read_csv('Files/error.csv')

for spectrum in spectra['file'].tolist():
    try:
        spec = Spectrum(spectrum)
        fig, ax = plt.subplots()
        spec.plotFlux(ax = ax)
        ax.set_xlabel('Wavelength \ Angstrom')
        ax.set_ylabel('Flux')
        ax.set_title(spectrum[:-5])
        plt.show()
    except:
        print('error opening ' + spectrum)
