import tensorflow as tf
import numpy as np
import glob as glob
from astropy.io import fits

class Neural_Net():
    def __init__(self, files):
        classes = ['STAR', 'GALAXY', 'QSO', 'Unknown']
        cls_length = len(classes)
        self.flux = []
        self.spec_class = []
        for idx, file in enumerate(files):
            with fits.open(file) as hdulist:
                self.flux.append((hdulist[0].data)[0])
                cls = hdulist[0].header['CLASS']
            self.spec_class.append([0]*cls_length)
            self.spec_class[-1][classes.index(cls)]=1
            
    def predict_classes():    
        


            
if __name__ == "__main__":
    sdir = '/data2/cpb405/DR1/'
    files = glob.glob(sdir + '*.fits')
    NN = Neural_Net(files)