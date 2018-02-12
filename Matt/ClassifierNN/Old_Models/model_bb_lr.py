from plot_model import plot_results

import tensorflow as tf
import numpy as np

import time

def blackbody(T):
    wavelength = np.linspace(3000, 9000, 3001)
    h = 6.63e-34
    c = 3e8
    k = 1.38e-23
    spec = ((8*np.pi*h*c)/((wavelength*1e-10)**5*(np.exp(h*c/((wavelength*1e-10)*k*T))-1)))/(T*55.8)
    return spec/np.sum(spec)
	#calculate a blackboduy curve for a given temperature

def line():
    wavelength = np.linspace(3000, 9000, 3001)
    ends = np.random.random(2)
    m = (ends[1] - ends[0])/(wavelength[-1] - wavelength[0])
    c = ends[0] - m*wavelength[0]
    line = (m*wavelength + c)
    return line/np.sum(line)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

#classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'Other']
classes = ['bb', 'Other']  

cls = len(classes)

samples = 10000
line_frac = 1/cls

train_frac = 0.7
batch_frac= 1./70.

record = 100
train_steps = 1000000

temps = np.random.uniform(1000, 10000, samples)
	#generate a random distribution of tempratures

temps[np.random.random(len(temps))<=line_frac] = np.nan
   
spectra = []
label = []

print('generating data...')

for t in temps:
    hot = [0]*cls
    if t!=t:
        spectrum = line()
        hot[-1] = 1
    else:
        spectrum = blackbody(t)
        #hot[int(t/1000) - 1] = 1
        hot[0] = 1
    label.append(hot)
    spectra.append(spectrum)

wavelengths = len(spectra[0])

spectra = np.array(spectra)
label = np.array(label)

split = np.random.random(samples)<=train_frac

x_train = np.array(spectra[split])
x_test = np.array(spectra[[not s for s in split]])

y_train = np.array(label[split])
y_test = np.array(label[[not s for s in split]])

print('data created\ntraining network')

x = tf.placeholder(tf.float32, shape = [None, wavelengths])
y_ = tf.placeholder(tf.float32, shape = [None, cls])

W = weight_variable([wavelengths, cls])
b = bias_variable([cls])

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
    for i in range(train_steps + 1):
        batch = np.random.random(len(x_train))<=batch_frac
        batch_x = x_train[batch]
        batch_y = y_train[batch]
        if i%record == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            print('step {} training accuracy {}'.format(i, train_accuracy))
            accuracies.append([i, train_accuracy])
        train_step.run(feed_dict={x: batch_x, y_: batch_y})

    conf, acc = sess.run([confusion, accuracy], feed_dict={x: x_test, y_: y_test})
    print('test accuracy {}'.format(acc))
    print(conf)
    np.savetxt('Files/bb_lr/classes.csv', classes, fmt = '%s', delimiter = ',')
    np.savetxt('Files/bb_lr/confusion.csv', conf, fmt = '%i', delimiter = ',')
    np.savetxt('Files/bb_lr/accuracies.csv', accuracies, delimiter = ',')
    print('training time: ', time.time() - t, 's')

plot_results('bb_lr')