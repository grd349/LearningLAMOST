from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open('../Shared/spec-55862-B6212_sp08-147.fits')

hdulist.info()

data = hdulist[0].data

plt.plot(data[1])

plt.show()

hdulist.close()

