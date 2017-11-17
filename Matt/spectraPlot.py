
from fits import Spectrum

path = '/data2/mrs493/DR1/'

spectra = ['spec-55862-B6212_sp06-145.fits',
           'spec-55862-B6212_sp06-117.fits',
           'spec-55860-B6001_sp06-145.fits',
           'spec-55862-B6202_sp06-145.fits',
           'spec-55862-B6202_sp06-117.fits']

for spectrum in spectra:
    spec = Spectrum(path + spectrum)
    spec.plotFlux()
    plt.show()