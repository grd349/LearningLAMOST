from plot_model import plot_results

import tensorflow as tf
import numpy as np

import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("TensorFlow/MNIST_data/", one_hot=True)

samples = 5000
batch_frac= 0.02
record = 100
train_steps = 10000

x_train, y_train = mnist.train.next_batch(samples)
x_test = mnist.test.images
y_test = mnist.test.labels

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cls = len(classes)
wavelengths = len(x_train[0])

x = tf.placeholder(tf.float32, shape = [None, wavelengths])
y_ = tf.placeholder(tf.float32, shape = [None, cls])

W = tf.Variable(tf.zeros([wavelengths,cls]))
b = tf.Variable(tf.zeros([cls]))

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
        batch = np.random.random(len(x_train))>=batch_frac
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
    np.savetxt('Files/MNIST_lr/classes.csv', classes, fmt = '%s', delimiter = ',')
    np.savetxt('Files/MNIST_lr/confusion.csv', conf, fmt = '%i', delimiter = ',')
    np.savetxt('Files/MNIST_lr/accuracies.csv', accuracies, delimiter = ',')
    print('training time: ', time.time() - t, 's')

plot_results('MNIST_lr')