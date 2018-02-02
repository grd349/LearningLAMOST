import tensorflow as tf
import numpy as np

class Neural_Net():
    def __init__(self):
        nStars = 1000
        nGalaxies = 1000       
        stars = []
        galaxies = []
        temps = np.random.uniform(3000,10000,nStars)
        wavelength = np.linspace(4000,9000,1000)
        for idx in range(len(temps)):
            stars.append(self.blackbody(temps[idx], wavelength))
        y1 = np.random.uniform(0,10000,nGalaxies)
        y2 = np.random.uniform(0,10000,nGalaxies)
        for idx in range(len(y1)):
            galaxies.append(np.linspace(y1[idx],y2[idx],1000))
        stars = np.array(stars)
        galaxies = np.array(galaxies)
        self.data = np.concatenate((stars,galaxies))
        print(self.data)
            
    def predict_class(self):
        #Model parameters
        W = tf.Variable([.3], dtype = tf.float32)
        b = tf.Variable([-.3], dtype = tf.float32)
        
        #Model input and output
        x = tf.placeholder(tf.float32)
        linear_model = W*x + b
        y = tf.placeholder(tf.float32)
        
        #Loss function (sum of the square error)
        loss = tf.reduce_sum(tf.square(linear_model - y))
        
        #Optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)
        
        #Training data
        x_train, x_test, y_train, y_test = train_test_split(self.df,test_size=0.5)
        y_train = [0,-1,-2,-3]
        data = {x: x_train, y: y_train}
        
        #Training loop
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init) #initialise model parameters
        for i in range(1000):
            sess.run(train, data)
        
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], data)
        print('W: {}, b: {}, loss: {}'.format(curr_W, curr_b, curr_loss))
        
    def blackbody(self, T, wavelength):
        ''' Models an ideal blackbody curve of a given temperature '''
        h = 6.63e-34
        c = 3e8
        k = 1.38e-23
        E = (8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1))
        return E

        
if __name__ == "__main__":
    NN = Neural_Net()
    NN.plot()