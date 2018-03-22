from plot_model import plot_results

import pandas as pd
import numpy as np
from astropy.io import fits
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt

import glob
import time

import tensorflow as tf

'''
CLASS --- DR3 --- TRAINING
A --- 275405 --- 1123
B --- 4939 ---  249
Carbon --- 1656 --- 107
DoubleStar --- 3401 --- 114
EM --- 183 --- 100
F --- 1269969 --- 4878
G --- 2293761 --- 9643
Galaxy --- 61815 --- 9726
K --- 1089096 --- 4205
M --- 319957 --- 1057
O --- 193 --- 100
QSO --- 16351 --- 9273
Unknown --- 408273 --- 8122
WD --- 9855 --- 100

T_Star --- 5268415 --- 21676
'''

class Neural_Network:
    def __init__(self):
        pass
    
    def weight_variable(self, shape):
        'create a variable of the specified shape, whith values initialised using a normal distribution with sigma = 0.1'
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
        
    def bias_variable(self, shape):
        'create a 1d variable of the specified length, initialised with the value 0.1'
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)
        
    def conv(self, x,W):
        'carry out a 1d convolution of the given matrix, moving by one element each time and padding with 0s to conserve length'
        return tf.nn.conv1d(x, W, 1,'SAME')
        
    def max_pool(self, x, width):
        'split x into blocks of size width, and replace those blocks with the max value in them'
        return tf.nn.pool(x, [width], 'MAX', 'SAME', strides = [width])
    
    def blackbody(self, T):
        'create a black body spectrum at temperature t and normalise it'
        wavelength = np.linspace(3000, 9000, 3001)
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        spec = ((8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1)))/(T*55.8)
        return self.noise(spec/np.sum(spec))
    
    def line(self):
        'create a line spectrum and normalise it'
        wavelength = np.linspace(3000, 9000, 3001)
        ends = np.random.random(2)
        m = (ends[1] - ends[0])/(wavelength[-1] - wavelength[0])
        c = ends[0] - m*wavelength[0]
        line = (m*wavelength + c)
        return self.noise(line/np.sum(line))
    
    def noise(self, spectrum):
        r = np.random.uniform(10, 240)
        sp = [np.random.normal(s*r, np.sqrt(s*r)) for s in spectrum]
        return sp

    def hot_v(self, t):
        'take a temperature, and return a hot vector for the MK class of a star at that temp (Nan goes to "Other")'
        hot = [0]*8
        if t>= 30000: hot[0] = 1
        elif t>=10000: hot[1] = 1
        elif t>=7500: hot[2] = 1
        elif t>=6000: hot[3] = 1
        elif t>=5200: hot[4]= 1
        elif t>=3700: hot[5] = 1
        elif t>=2400: hot[6] = 1
        else: hot[-1] = 1
        return hot

    def train_test_split(self, train_frac):
        'split the data into a train and a test set, with train_frac of the points in the training set'
        split = np.random.random(self.samples)<=train_frac

        self.x_train = np.array(self.spectra[split])
        self.x_test = np.array(self.spectra[[not s for s in split]])
        
        self.y_train = np.array(self.label[split])
        self.y_test = np.array(self.label[[not s for s in split]])
        
        self.file_train = np.array(self.files[split])
        self.file_test = np.array(self.files[[not s for s in split]])
        
    def batch(self, batch_frac):
        'return a batch of size batch_frac relative to the train set'
        batch = np.random.random(len(self.x_train))<=batch_frac
        batch_x = self.x_train[batch]
        batch_y = self.y_train[batch]
        return batch_x, batch_y
    
    def save(self, folder, conf, accuracies, w1, w2):
        'save the results of the model to .csv files'
        np.savetxt('Files/' + folder + '/classes.csv', self.classes, fmt = '%s', delimiter = ',')
        np.savetxt('Files/' + folder + '/confusion.csv', conf, fmt = '%i', delimiter = ',')
        np.savetxt('Files/' + folder + '/accuracies.csv', accuracies, delimiter = ',')

        with open('Files/' + folder + '/w1.csv', 'w') as outfile:
            outfile.write('#Layer 1\n')
            for flter in w1:
                np.savetxt(outfile, flter, fmt = '%s', delimiter = ',')
                outfile.write('#New filter\n')
        with open('Files/' + folder + '/w2.csv', 'w') as outfile:
            outfile.write('#Layer 2\n')
            for flter in w2:
                np.savetxt(outfile, flter, fmt = '%s', delimiter = ',')
                outfile.write('#New filter\n')

    def make_spectra(self, samples, line_frac): 
        'produce a test data set of samples spectra, of which line_frac are lines and the remained are blackbodies with a uniform distribution of temperatures'
        self.samples = samples
        
        ti = time.time()
        print('generating data...')
        
        self.classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'Other']
        
        self.cls = len(self.classes)
        
        temps = np.random.beta(2,6,samples)*10000 + 2400
        
        #temps = np.random.uniform(2400, 33000, samples)
        
        temps[np.random.random(len(temps))<=line_frac] = np.nan
        
        spectra = []
        label = []
        
        for t in temps:
            if t!=t: spectrum = self.line()
            else: spectrum = self.blackbody(t)
            hot = self.hot_v(t)
            label.append(hot)
            spectra.append(spectrum)
        
        self.wavelengths = len(spectra[0])
        
        self.spectra = np.array(spectra)
        self.label = np.array(label)
        
        print('data generated: ', time.time() - ti)

    def get_LAMOST(self, Ldir, MK = False, SNR = 0):
        'read in the spectra from the LAMOST data'
        
        ti = time.time()
        print('reading data...')
        
        train_files = glob.glob(Ldir)
        
        flux = []
        CLASS = []
        files = []
        
        self.wavelengths = 3500
        
        for idx, file in enumerate(train_files):
            with fits.open(file) as hdulist:
                flx = hdulist[0].data[0]
                flx = flx[:self.wavelengths]
                flx = flx/np.sum(flx)
                CLS = hdulist[0].header['CLASS']
                fn = hdulist[0].header['FILENAME']
                if MK and CLS=='STAR': CLS = hdulist[0].header['SUBCLASS'][0]
                try:
                    U = hdulist[0].header['SNRU']
                    G = hdulist[0].header['SNRG']
                    R = hdulist[0].header['SNRR']
                    I = hdulist[0].header['SNRI']
                    Z = hdulist[0].header['SNRZ']
                except:
                    U = hdulist[0].header['SN_U']
                    G = hdulist[0].header['SN_G']
                    R = hdulist[0].header['SN_R']
                    I = hdulist[0].header['SN_I']
                    Z = hdulist[0].header['SN_Z']
            if U>=SNR or G>=SNR or R>=SNR or I>=SNR or Z>=SNR:
                flux.append(flx)
                CLASS.append(CLS)
                files.append(fn)
        
        le = LabelEncoder()
        CLAS = le.fit_transform(CLASS)
        
        self.classes = le.inverse_transform(np.arange(np.amax(CLAS)+1))
        
        self.cls = len(self.classes)
        
        enc = OneHotEncoder(sparse=False)
        CLA = enc.fit_transform(CLAS.reshape(-1,1))
        
        ###        
        
        self.samples = len(flux)        
        
        self.spectra = np.array(flux)
        self.label = np.array(CLA)
        self.files = np.array(files)
        
        print('data read: ', time.time() - ti, '\ndata contents:')
        
        for i in range(self.cls):
            print(self.classes[i], ': ', np.sum([x[i] for x in CLA]))
            
        ti = time.time()
        
    def get_LAMOST_tt(self, train_dir, test_dir, MK = False):
        'read in the spectra from the LAMOST data'
        
        ti = time.time()
        print('reading training data...')
        
        train_files = glob.glob(train_dir)
        
        flux = []
        CLASS = []
        
        self.wavelengths = 3500
        
        for idx, file in enumerate(train_files):
            with fits.open(file) as hdulist:
                flx = hdulist[0].data[0]
                flx = flx[:self.wavelengths]
                flx = flx/np.sum(flx)
                CLS = hdulist[0].header['CLASS']
                if MK and CLS=='STAR': CLS = hdulist[0].header['SUBCLASS'][0]
            flux.append(flx)
            CLASS.append(CLS)
        
        le = LabelEncoder()
        CLAS = le.fit_transform(CLASS)
        
        self.classes = le.inverse_transform(np.arange(np.amax(CLAS)+1))
        
        self.cls = len(self.classes)
        
        enc = OneHotEncoder(sparse=False)
        CLA = enc.fit_transform(CLAS.reshape(-1,1))
        
        self.x_train = np.array(flux)
        self.y_train = np.array(CLA)
        
        print('training data read: ', time.time() - ti, '\ntraining data contents:')
        
        for i in range(self.cls):
            print(self.classes[i], ': ', np.sum([x[i] for x in CLA]))
            
        ti = time.time()
        
        print('reading test data...')
        
        test_files = glob.glob(test_dir)
        
        flux2 = []
        CLASS2 = []
        
        for idx, file in enumerate(test_files):
            with fits.open(file) as hdulist:
                flx = hdulist[0].data[0]
                flx = flx[:self.wavelengths]
                flx = flx/np.sum(flx)
                CLS = hdulist[0].header['CLASS']
                if MK and CLS=='STAR': CLS = hdulist[0].header['SUBCLASS'][0]
            flux2.append(flx)
            CLASS2.append(CLS)
                    
        CLAS2 = le.transform(CLASS2)
        CLA2 = enc.transform(CLAS2.reshape(-1,1))
        
        self.x_test = np.array(flux2)
        self.y_test = np.array(CLA2)
        
        print('test data read: ', time.time() - ti, '\ntest data contents:')
        
        for i in range(self.cls):
            print(self.classes[i], ': ', np.sum([x[i] for x in CLA2]))
        
    def train_lr(self, folder, train_steps, batch_frac, record):
        'create a linear regressor neural net and train it on the data'
        x = tf.placeholder(tf.float32, shape = [None, self.wavelengths])
        y_ = tf.placeholder(tf.float32, shape = [None, self.cls])
        
        W = self.weight_variable([self.wavelengths, self.cls])
        b = self.bias_variable([self.cls])
        
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        accuracies = []
        
        confusion = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(y_,1))
        
        with tf.Session() as sess:
            t = time.time()
            sess.run(tf.global_variables_initializer())
            for i in range(train_steps):
                batch_x, batch_y = self.batch(batch_frac)
                if i%record == 0 and i != 0:
                    train_accuracy = sess.run(accuracy, feed_dict={x: self.x_test, y_: self.y_test})
                    print('step {} training accuracy {}'.format(i, train_accuracy))
                    accuracies.append([i, train_accuracy])
                train_step.run(feed_dict={x: batch_x, y_: batch_y})
        
            conf, acc = sess.run([confusion, accuracy], feed_dict={x: self.x_test, y_: self.y_test})
            print('test accuracy {}'.format(acc))
            print(conf)
            accuracies.append([i+1, acc])
            self.save(folder, conf, accuracies)
            print('training time: ', time.time() - t, 's')
        
        plot_results(folder)
        
    def train_conv(self, folder, train_steps, batch_frac, keep=0.5, record=100, pw0=3, pw1=10, pw2=10, width1=50, width2=50, inter1=32, inter2=64, inter3=1000):
        'create a convolutional neural net and train it on the data'
        
        f_wavs = self.wavelengths

        for pw in [pw0, pw1, pw2]:
            f_wavs = int(np.ceil(f_wavs/pw))
        
        x = tf.placeholder(tf.float32, shape = [None, self.wavelengths])
        y_ = tf.placeholder(tf.float32, shape = [None, self.cls])
        
        
        
        i_l1 = tf.reshape(x, [-1, self.wavelengths, 1])
        
        m_l1 = self.max_pool(i_l1, pw0)
        
        W_l1 = self.weight_variable([width1, 1,inter1])
        b_l1 = self.bias_variable([inter1])
        
        o_l1 = tf.nn.relu(self.conv(m_l1, W_l1) + b_l1)
        
        
        
        i_l2 = self.max_pool(o_l1, pw1)
        
        W_l2 = self.weight_variable([width2, inter1,inter2])
        b_l2 = self.bias_variable([inter2])
        
        o_l2 = tf.nn.relu(self.conv(i_l2, W_l2) + b_l2)
        
        
        
        i_l3 = self.max_pool(o_l2, pw2)
        
        m_l3 = tf.reshape(i_l3, [-1, f_wavs*inter2])
        
        W_l3 = self.weight_variable([f_wavs*inter2, inter3])
        b_l3 = tf.Variable(tf.zeros([inter3]))
        
        o_l3 =  tf.nn.relu(tf.matmul(m_l3, W_l3) + b_l3)
        
        
        
        keep_prob= tf.placeholder(tf.float32)
        i_l4 = tf.nn.dropout(o_l3, keep_prob)
        
        W_l4 = self.weight_variable([inter3, self.cls])
        b_l4 = self.bias_variable([self.cls])
        
        y = tf.matmul(i_l4, W_l4) + b_l4
        
        
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
        
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        accuracies = []
        
        confusion = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(y_,1))
        
        
        with tf.Session() as sess:
            t = time.time()
            ti =time.time()
            sess.run(tf.global_variables_initializer())
            for i in range(train_steps):
                batch_x, batch_y = self.batch(batch_frac)
                if i%record == 0 and i != 0:
                    train_accuracy = sess.run(accuracy, feed_dict={x: self.x_test, y_: self.y_test, keep_prob: 1.0})
                    print('step {} training accuracy {}, {}s'.format(i, train_accuracy, time.time() - ti))
                    ti = time.time()
                    accuracies.append([i, train_accuracy])
                train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: keep})
            conf, acc, filter1, filter2, probs = sess.run([confusion, accuracy, W_l1, W_l2, y], feed_dict={x: self.x_test, y_: self.y_test, keep_prob: 1.0})
            print('test accuracy {}'.format(acc))
            print(conf)
            accuracies.append([i+1, acc])
            self.save(folder, conf, accuracies, filter1, filter2)
            print('training time: ', time.time() - t, 's')
        
        plot_results(folder)

        for i in range(len(self.file_test)):
            if self.y_test[i][3] and np.argmax(probs[i])==1:
                print(self.file_test[i])
                fig, ax = plt.subplots()
                ax.plot(self.x_test[i])
                plt.show()

if __name__ == "__main__":
    train = '/data2/cpb405/Training_2/*.fits'
    test = '/data2/mrs493/DR1_3/*.fits'  
    
    train = '/data2/cpb405/Training/*.fits'
    
    NN = Neural_Network()
    MK = False
    #samples = 10000
    tts = 0.5
    bf = 0.01
    SNR = 0
    #NN.make_spectra(samples, 1./8.)
    #NN.get_LAMOST_tt(train, test, MK)    
    NN.get_LAMOST(train, MK, SNR)
    NN.train_test_split(tts)
    conv = {'folder':'test', 'train_steps':10000, 'batch_frac':bf, 'keep':0.5, 'record':100, 'pw0':4, 'pw1':10, 'pw2':10, 'width1':50, 'width2':50, 'inter1':32, 'inter2':64, 'inter3':1000}
    NN.train_conv(**conv)
    
