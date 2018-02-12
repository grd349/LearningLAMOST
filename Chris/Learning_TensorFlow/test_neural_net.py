from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import numpy as np
import tensorflow as tf

class Neural_Net():
    def __init__(self):
        self.MKClasses = ['B-type','A-type','F-type','G-type','K-type','M-type']
        self.MKTemps = [10000,7500,6000,5200,3700,2400]
    
    def read_lamost_data(self,files):
        ''' Reads in the flux and classes from LAMOST fits files & converts classes to one-hot vectors '''
        self.flux = []
        self.scls = []
        for idx, file in enumerate(files):
            with fits.open(file) as hdulist:
                self.flux.append((hdulist[0].data)[0])
                self.scls.append(hdulist[0].header['CLASS'])
        self.cls = self.onehot(self.scls)
        
    def create_artificial_data(self,nStars,nGalaxies,plot=True):
        ''' Creates a set of artificial stars (modelled as blackbodies) and galaxies (modelled as straight lines) '''
        fStar = []
        cStar = []
        temps = np.random.uniform(2400,15000,nStars)
        wavelength = np.linspace(4000,9000,1000)
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
        
        if plot == True:
            for idx in range(10):
                fig, ax = plt.subplots()
                ax.plot(wavelength,fStar[idx])
                ax.plot(wavelength,fGalaxy[idx])
                ax.set_xlabel("Wavelength / Angstroms")
                ax.set_ylabel("Flux")

        self.flux = np.concatenate((fStar,fGalaxy))
        self.scls = np.concatenate((cStar,cGalaxy))
        self.cls = self.onehot(self.scls)
        
    def predict_class(self):
        ''' Sets up a neural net with one layer using a simple linear estimator '''
        x = tf.placeholder(tf.float32, [None, len(self.flux[0])])
        
        W = tf.Variable(tf.truncated_normal([len(self.flux[0]),len(self.cls[0])],stddev=1./np.sqrt(len(self.flux[0]))), dtype = tf.float32)
        b = tf.Variable(np.zeros(len(self.cls[0])), dtype = tf.float32)
        
        y = tf.nn.softmax(tf.matmul(x, W) + b)      
        y_ = tf.placeholder(tf.float32, [None, len(self.cls[0])])
        
        x_train, x_test, y_train, y_test = train_test_split(self.flux, self.cls, test_size=0.5)

        print(y_test)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)           
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                batch = np.random.random(len(x_train)) > 0.01
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                sess.run(train_step, feed_dict={x: x_train[batch], y_: y_train[batch]})
                        
            print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
    
    def convolution(self):
        ''' Sets up a 1D convolutional multi-layered neural net '''
        x_train, x_test, y_train, y_test, labels_train, self.labels = train_test_split(self.flux, self.cls, self.scls, test_size=0.5)
        
        x = tf.placeholder(tf.float32, [None, len(self.flux[0])])
        x2 = tf.reshape(x,[-1,len(self.flux[0]),1])
         
        y_ = tf.placeholder(tf.float32, [None, len(self.cls[0])])
        
        W_conv1 = self.weight_variable([500,1,32])
        b_conv1 = self.bias_variable([32])
        
        h_conv1 = tf.nn.relu(self.conv1d(x2, W_conv1) + b_conv1)
        """
        W_conv2 = self.weight_variable([500,32,64])
        b_conv2 = self.bias_variable([64])
        
        h_conv2 = tf.nn.relu(self.ctensorflow plot predictions from convolutiononv1d(h_conv1, W_conv2) + b_conv2)
        """
        W_fc1 = self.weight_variable([len(self.flux[0]) * 32, 1024])
        b_fc1 = self.bias_variable([1024])
        
        h_pool2_flat = tf.reshape(h_conv1, [-1, len(self.flux[0]) * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        W_fc2 = self.weight_variable([1024, 7])
        b_fc2 = self.bias_variable([7])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
       
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))         
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.accuracy = []
        self.batch = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(2500):
                batch = np.random.random(len(x_train)) > 0.01
                if i % 10 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    self.accuracy.append(train_accuracy)
                    self.batch.append(i)
                train_step.run(feed_dict={x: x_train[batch], y_: y_train[batch], keep_prob: 0.5})         
            print('test accuracy %g' % accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))
            self.correct_pred = correct_prediction.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    
    def save(self):
        ''' Saves final results from neural net into csv files '''
        np.savetxt('AccBatch.csv', np.column_stack((self.batch,self.accuracy)), delimiter=',')
        np.savetxt('Predictions.csv', self.correct_pred, delimiter=',')
        np.savetxt('Labels.csv', self.labels, delimiter=',', fmt="%s")
        
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
            E[idx] = E[idx] + np.random.normal(0,np.sqrt(E[idx]/1000))
        return E/np.sum(E)
    
    def onehot(self, classes):
        ''' Encodes a list of descriptive labels as one hot vectors '''
        label_encoder = LabelEncoder()
        int_encoded = label_encoder.fit_transform(classes)
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
        
if __name__ == "__main__":
    sdir = '/data2/mrs493/DR1_3/'
    files = glob.glob(sdir + '*.fits')
    
    NN = Neural_Net()
    NN.read_lamost_data(files)
    #NN.create_artificial_data(1000,100,False)
    NN.convolution()
    NN.save()