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

def conv(x,W):
	return tf.nn.conv1d(x, W, 1,'SAME')

def max_pool(x, width):
    return tf.nn.pool(x, [width], 'MAX', 'SAME', strides = [width])

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'Other']
#classes = ['bb', 'Other']  

cls = len(classes)

samples = 1000
line_frac = 1/cls

train_frac = 0.7
batch_frac= 1./70.

pw0 = 3

width1 = 50
inter1 = 32
pw1 = 10

width2 = width1
inter2 = 2*inter1
pw2 = 10

inter3 = 1000

keep = 0.5

record = 100
train_steps = 30000

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
        hot[int(t/1000) - 1] = 1
        #hot[0] = 1
    label.append(hot)
    spectra.append(spectrum)

wavelengths = len(spectra[0])

f_wavs = wavelengths

for pw in [pw0, pw1, pw2]:
    f_wavs = int(np.ceil(f_wavs/pw))

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



i_l1 = tf.reshape(x, [-1, wavelengths, 1])

m_l1 = max_pool(i_l1, pw0)

W_l1 = weight_variable([width1, 1,inter1])
b_l1 = bias_variable([inter1])

o_l1 = tf.nn.relu(conv(m_l1, W_l1) + b_l1)



i_l2 = max_pool(o_l1, pw1)

W_l2 = weight_variable([width2, inter1,inter2])
b_l2 = bias_variable([inter2])

o_l2 = tf.nn.relu(conv(i_l2, W_l2) + b_l2)



i_l3 = max_pool(o_l2, pw2)

m_l3 = tf.reshape(i_l3, [-1, f_wavs*inter2])

W_l3 = weight_variable([f_wavs*inter2, inter3])
b_l3 = tf.Variable(tf.zeros([inter3]))

o_l3 =  tf.nn.relu(tf.matmul(m_l3, W_l3) + b_l3)



keep_prob= tf.placeholder(tf.float32)
i_l4 = tf.nn.dropout(o_l3, keep_prob)

W_l4 = weight_variable([inter3, cls])
b_l4 = bias_variable([cls])

y = tf.matmul(i_l4, W_l4) + b_l4


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracies = []

confusion = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(y_,1))

with tf.Session() as sess:
    t = time.time()
    sess.run(tf.global_variables_initializer())
    for i in range(train_steps):
        batch = np.random.random(len(x_train))<=batch_frac
        batch_x = x_train[batch]
        batch_y = y_train[batch]
        if i%record == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
            print('step {} training accuracy {}'.format(i, train_accuracy))
            accuracies.append([i, train_accuracy])
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: keep})
    conf, acc = sess.run([confusion, accuracy], feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    print('test accuracy {}'.format(acc))
    print(conf)
    accuracies.append([i+1, acc])
    np.savetxt('Files/bb_conv/classes.csv', classes, fmt = '%s', delimiter = ',')
    np.savetxt('Files/bb_conv/confusion.csv', conf, fmt = '%i', delimiter = ',')
    np.savetxt('Files/bb_conv/accuracies.csv', accuracies, delimiter = ',')
    print('training time: ', time.time() - t, 's')

plot_results('bb_conv')