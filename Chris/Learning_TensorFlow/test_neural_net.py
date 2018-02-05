import numpy as np
import tensorflow as tf

class Neural_Net():
    def __init__(self):
        nStars = 1000
        nGalaxies = 1000       
        fStar = []
        cStar = []
        fGalaxy = []
        cGalaxy = []
        temps = np.random.uniform(3000,10000,nStars)
        wavelength = np.linspace(4000,9000,1000)
        for idx in range(len(temps)):
            fStar.append(self.blackbody(temps[idx], wavelength))
            cStar.append([1,0])
        fStar = np.array(fStar)
        cStar = np.array(cStar)

        y1 = np.random.uniform(0,10000,nGalaxies)
        y2 = np.random.uniform(0,10000,nGalaxies)
        for idx in range(len(y1)):
            fGalaxy.append(np.linspace(y1[idx],y2[idx],1000))
            cGalaxy.append([0,1])
        fGalaxy = np.array(fGalaxy)
        cGalaxy = np.array(cGalaxy)

        self.flux = np.concatenate((fStar,fGalaxy))
        self.cls = np.concatenate((cStar,cGalaxy))
        self.sample = np.random.random(nStars+nGalaxies) > 0.7
            
    def predict_class(self):
        
        x = tf.placeholder(tf.float32, [None, len(self.flux[0])])
        
        W = tf.Variable(np.zeros([len(self.flux[0]),len(self.cls[0])]), dtype = tf.float32)
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
            for i in range(1000):
                batch = np.random.random(len(x_train)) > 0.01
                if i % 100 == 0:
                    #train_accuracy = accuracy.eval(feed_dict={x: x_train[batch], y_: y_train[batch]})
                    train_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                sess.run(train_step, feed_dict={x: x_train[batch], y_: y_train[batch]})
                        
            print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
        
    def blackbody(self, T, wavelength):
        ''' Models an ideal blackbody curve of a given temperature '''
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        E = (8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))
        return E

        
if __name__ == "__main__":
    NN = Neural_Net()
    NN.predict_class()