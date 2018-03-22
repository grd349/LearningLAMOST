from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from astropy.io import fits
import glob
import time
import numpy as np
import tensorflow as tf

"""
CLASS --- TOTAL --- TRAINING
A --- 275405 --- 1123
B --- 4939 --- 712
Carbon --- 1656 --- 706
DoubleStar --- 3401 --- 1012
EM --- 183 --- 183
F --- 1269969 --- 4878
G --- 2293761 --- 9643
Galaxy --- 61815 --- 9726
K --- 1089096 --- 4205
M --- 319957 --- 1057
O --- 193 --- 193
QSO --- 16351 --- 9273
Star --- 5268415 --- 25412
Unknown --- 408273 --- 8122
WD --- 9855 --- 1700
"""
class Neural_Net():
    def __init__(self):
        pass
    
    def read_lamost_data(self,sfile,MK = False):
        ''' Reads in the flux and classes from LAMOST fits files & converts classes to one-hot vectors '''
        print("Reading in LAMOST data...")
        flux = []
        scls = []
        self.wav = 3500
        for idx, file in enumerate(sfile):
            with fits.open(file) as hdulist:
                f = hdulist[0].data[0]
                f = f[:self.wav]
                f = f/np.sum(f)
                s = hdulist[0].header['CLASS']
                if MK and s == 'STAR':
                    s = hdulist[0].header['SUBCLASS'][0]
            flux.append(f)
            scls.append(s)
            
        class_dict = {}
        for s in scls:
            if s in class_dict:
                class_dict[s] += 1
            else:
                class_dict[s] = 1
        print(class_dict)
        
        flux = np.array(flux)
        scls = np.array(scls)
        cls = self.onehot(scls)
        
        self.fluxTR, self.fluxTE, self.clsTR, self.clsTE = train_test_split(flux, cls, test_size=0.5)
        print("LAMOST data successfully read in...")
        
    def create_artificial_data(self,nStars,nGalaxies):
        ''' Creates a set of artificial stars (modelled as blackbodies) and galaxies (modelled as straight lines) '''
        print("Generating artificial data...")
        self.MKClasses = ['B-type','A-type','F-type','G-type','K-type','M-type']
        self.MKTemps = [10000,7500,6000,5200,3700,2400]
        fStar = []
        cStar = []
        self.wav = 1000
        temps = np.random.uniform(2400,15000,nStars)
        wavelength = np.linspace(4000,9000,self.wav)
        for idx in range(nStars):
            flux, cls = self.blackbody(temps[idx], wavelength)
            fStar.append(flux)
            cStar.append(cls)
        fStar = np.array(fStar)
        cStar = np.array(cStar)

        fGalaxy = []
        cGalaxy = []
        for idx in range(nGalaxies):
            fGalaxy.append(self.line())
            cGalaxy.append('Galaxy')
        fGalaxy = np.array(fGalaxy)
        cGalaxy = np.array(cGalaxy)

        self.flux = np.concatenate((fStar,fGalaxy))
        self.scls = np.concatenate((cStar,cGalaxy))
        self.cls = self.onehot(self.scls)

        self.fluxTR, self.fluxTE, self.clsTR, self.clsTE = train_test_split(self.flux, self.cls, test_size=0.5)
        print("Artificial data created.")
    
    def convolution(self, steps, pool_width=15):
        ''' Sets up a 1D convolutional multi-layered neural net '''
        print("Performing 1D convolution...")
        n_classes = len(self.clsTR[0])
        x = tf.placeholder(tf.float32, [None, self.wav])
        x_ = tf.reshape(x,[-1,self.wav,1])  
        y_ = tf.placeholder(tf.float32, [None, n_classes])
        
        ''' First Pooling Layer '''
        ''' This initial layer of pooling reduces the 3500 flux values in each input spectrum to 234 to speed up computation of first convolution. '''
        h_pool1 = self.max_pool(x_, pool_width)
        
        ''' First Convolutional Layer '''
        ''' Convolves patches of 50 points with 32 weight filters (each of width 50), increasing the number of features per flux value from 1 to 32. '''
        ''' The input is zero padded before convolution so that the number of neurons before and after this operation is the same. '''
        W_conv1 = self.weight_variable([50,1,32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv1d(h_pool1, W_conv1) + b_conv1)
        
        ''' Second Pooling Layer '''
        ''' Now that each neuron contains 32 values rather than 1, a second layer of pooling is carried out, reducing the number of neurons to 16. '''
        h_pool2 = self.max_pool(h_conv1, pool_width)
        
        ''' Second Convolutional Layer '''
        ''' Convolves patches of 50 points with 64 weight filters (of size 50x32), increasing the number of features from 32 to 65. '''
        W_conv2 = self.weight_variable([50,32,64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv1d(h_pool2, W_conv2) + b_conv2)
        
        ''' Third Pooling Layer '''
        ''' Reduces number of neurons from 16 to 2. '''
        h_pool3 = self.max_pool(h_conv2, pool_width)
      
        ''' Densely Connected Layer '''
        ''' Flattens the previous layer to a batch of arrays (one for each input spectrum), multiplies by a weight matrix, adds a bias and applies a ReLU. '''
        ''' The number of neurons in the next layer is 1024 (although these each hold one value rather than 64). The ReLU zeros any negative values. '''
        dim = self.wav
        for pool in range(3):
            dim = int(np.ceil(dim/pool_width))
        h_pool3_flat = tf.reshape(h_pool3, [-1, dim * 64])
        W_fc1 = self.weight_variable([dim * 64, 1024])         
        b_fc1 = self.bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        
        ''' Dropout (to reduce overfitting) '''
        ''' Temporarily removes several neurons from the previous layer at random, in order to reduce overfitting. '''
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        ''' Readout Layer '''
        ''' Produces a layer of neurons equal to the number of classes, containing values corresponding to the probabilities that the input spectrum belongs '''
        ''' to each class. '''
        W_fc2 = self.weight_variable([1024, n_classes])
        b_fc2 = self.bias_variable([n_classes])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        ''' Train and Evaluate Model '''
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
       
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))         
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.accuracy = []
        confusion = tf.confusion_matrix(tf.argmax(y_conv,1), tf.argmax(y_,1))
        
        with tf.Session() as sess:
            t = time.time()
            sess.run(tf.global_variables_initializer())
            for i in range(steps):
                batch = np.random.random(len(self.fluxTR)) < 0.01
                if i % 50 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: self.fluxTE, y_: self.clsTE, keep_prob: 1.0})
                    print('Step %d, Training Accuracy %g' % (i, train_accuracy))
                    self.accuracy.append(train_accuracy)
                train_step.run(feed_dict={x: self.fluxTR[batch], y_: self.clsTR[batch], keep_prob: 0.5})         
            self.conf = sess.run(confusion, feed_dict={x: self.fluxTE, y_: self.clsTE, keep_prob: 1.0})
            print('Test Accuracy %g' % accuracy.eval(feed_dict={x: self.fluxTE, y_: self.clsTE, keep_prob: 1.0}))
            print('Time Taken: ', (time.time() - t)/3600, 'hours')
    
    def save(self,folder):
        ''' Saves final results from neural net into csv files '''
        print("Saving results...")
        np.savetxt('Files/' + folder + '/Accuracy.csv', np.column_stack(self.accuracy), delimiter=',')
        np.savetxt('Files/' + folder + '/Confusion.csv', self.conf, fmt='%i', delimiter=',')
        np.savetxt('Files/' + folder + '/Labels.csv', self.labels, fmt='%s', delimiter=',')
        
    def blackbody(self, T, wavelength):
        ''' Models an ideal blackbody curve of a given temperature '''
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        E = (8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))      
        for idx in range(len(E)):
            E[idx] = E[idx] + np.random.normal(0,np.sqrt(E[idx]*100))      
        for idx in range(len(self.MKTemps)):
            if T > self.MKTemps[idx]:
                cls = self.MKClasses[idx]
                break           
        return E/np.sum(E), cls
    
    def line(self):
        ''' Models a galaxy as a straight line '''
        y = np.abs(np.random.random(2))
        E = np.linspace(y[0],y[1],1000)
        for idx in range(len(E)):
            E[idx] = E[idx] + np.ranpoolsdom.normal(0,np.sqrt(E[idx]/1000))
        return E/np.sum(E)
    
    def onehot(self,classes):
        ''' Encodes a list of descriptive labels as one hot vectors '''
        label_encoder = LabelEncoder()
        int_encoded = label_encoder.fit_transform(classes)
        self.labels = label_encoder.inverse_transform(np.arange(np.amax(int_encoded)+1))
        onehot_encoder = OneHotEncoder(sparse=False)
        int_encoded = int_encoded.reshape(len(int_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(int_encoded)
        return onehot_encoded
    
    def weight_variable(self, shape):
        ''' Randomly initialises weight variable '''
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        ''' Randomly initialises bias variable '''
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv1d(self, x, W):
        ''' Performs 1D convolution '''
        return tf.nn.conv1d(x, W, 1, 'SAME')
    
    def max_pool(self, x, width):
        ''' Pools the data, reducing the dimensions by a factor of "width" '''
        return tf.nn.pool(x, [width], 'MAX', 'SAME', strides=[width])
        
if __name__ == "__main__":
    sdir = '/data2/cpb405/Training/'
    files = glob.glob(sdir + '*.fits')
    
    NN = Neural_Net()
    NN.read_lamost_data(files, MK=False)
    NN.convolution(steps=200000)
    NN.save('DR3_12')
