from fits import Spectrum

import glob

for fitsName in glob.glob('/data2/mrs493/DR1/*.fits')[:20]:
    spectra = Spectrum(fitsName)
    
    spectra.plotFlux()
    plt.xlim([5750, 5850])
    plt.show()
    