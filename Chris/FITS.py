import matplotlib.pyplot as plt
import glob
from astropy.io import fits

for fitsName in glob.glob('../Data/relearninglamost/*.fits'):
    hdulist = fits.open(fitsName)
    
    dat = hdulist[0].data

    date = hdulist[0].header["DATE"]
    t_ID = hdulist[0].header["T_INFO"]
    SNR = hdulist[0].header["SN_U"]

    plt.plot(dat[0])
    plt.xlabel("Wavelength [Angstroms]")
    plt.ylabel("Flux")
    plt.title("ID {}, SNR {}, Date {}".format(t_ID, SNR, date))
    plt.show()

    hdulist.close()

