import numpy as np
import tensorflow as tf

class Neural_Net():
    def __init__(self):
        nStars = 10000
        nGalaxies = 10000    
        fStar = []
        cStar = []
        fGalaxy = []
        cGalaxy = []
        temps = np.random.uniform(3000,10000,nStars)
        wavelength = np.linspace(4000,9000,1000)
        for idx in range(nStars):
            fStar.append(self.blackbody(temps[idx], wavelength))
            cStar.append([1,0])
        fStar = np.array(fStar)
        cStar = np.array(cStar)

        for idx in range(nGalaxies):
            fGalaxy.append(self.line())
            cGalaxy.append([0,1])
        fGalaxy = np.array(fGalaxy)
        cGalaxy = np.array(cGalaxy)

        self.flux = np.concatenate((fStar,fGalaxy))
        self.cls = np.concatenate((cStar,cGalaxy))
        self.sample = np.random.random(nStars+nGalaxies) > 0.7
            
    def predict_class(self):
        
        x = tf.placeholder(tf.float32, [None, len(self.flux[0])])
        
        #W = tf.Variable(np.zeros([len(self.flux[0]),len(self.cls[0])]), dtype = tf.float32)
        W = tf.Variable(tf.truncated_normal([len(self.flux[0]),len(self.cls[0])],stddev=1./np.sqrt(len(self.flux[0]))), dtype = tf.float32)
        b = tf.Variable(np.zeros(len(self.cls[0])), dtype = tf.float32)
        
        y = tf.nn.softmax(tf.matmul(x, W) + b)      
        y_ = tf.placeholder(tf.float32, [None, len(self.cls[0])])
        
        x_train = self.flux[self.sample]
        x_test = self.flux[[not s for s in self.sample]]
        y_train = self.cls[self.sample]
        y_test = self.cls[[not s for s in self.sample]]
        
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)           
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                batch = np.random.random(len(x_train)) > 0.01
                if i % 100 == 0:
                    #train_accuracy = accuracy.eval(feed_dict={x: x_train[batch], y_: y_train[batch]})
                    train_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                sess.run(train_step, feed_dict={x: x_train[batch], y_: y_train[batch]})
                        
            print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
    
    def convolution(self):
        x = tf.placeholder(tf.float32, [None, len(self.flux[0])])
        x2 = tf.reshape(x,[-1,len(self.flux[0]),1])
         
        y_ = tf.placeholder(tf.float32, [None, len(self.cls[0])])
        
        x_train = self.flux[self.sample]
        x_test = self.flux[[not s for s in self.sample]]
        y_train = self.cls[self.sample]
        y_test = self.cls[[not s for s in self.sample]]
        
        W_conv1 = self.weight_variable([500,1,32])
        b_conv1 = self.bias_variable([32])
        
        h_conv1 = tf.nn.relu(self.conv1d(x2, W_conv1) + b_conv1)
        
        W_conv2 = self.weight_variable([500,32,64])
        b_conv2 = self.bias_variable([64])
        
        h_conv2 = tf.nn.relu(self.conv1d(h_conv1, W_conv2) + b_conv2)
        
        W_fc1 = self.weight_variable([len(self.flux[0]) * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        
        h_pool2_flat = tf.reshape(h_conv2, [-1, len(self.flux[0]) * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        W_fc2 = self.weight_variable([1024, 2])
        b_fc2 = self.bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) 
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                batch = np.random.random(len(x_train)) > 0.01
                sess.run(x2, feed_dict={x: x_train[batch], y_: y_train[batch]})
                print('Lol sucks')
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
                    #print('step %d, training accuracy %g' % (i, train_accuracy))
                #train_step.run(feed_dict={x: x_train[batch], y_: y_train[batch], keep_prob: 0.5})
            print('test accuracy %g' % accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))
            
    def blackbody(self, T, wavelength):
        ''' Models an ideal blackbody curve of a given temperature '''
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        E = (8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))
        return E/np.sum(E)
    
    def line(self):
        ''' Models a galaxy as a straight line '''
        y = np.random.random(2)
        E = np.linspace(y[0],y[1],1000)
        return E/np.sum(E)
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv1d(self, x, W):
        return tf.nn.conv1d(x, W, 1, 'SAME')
        
if __name__ == "__main__":
    NN = Neural_Net()
    NN.convolution()