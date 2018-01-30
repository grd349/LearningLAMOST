import glob
from astropy.io import fits
import numpy as sp
import tensorflow as tf

files = glob.glob('/data2/mrs493/DR1_2/*.fits')

classes = ['STAR', 'GALAXY', 'QSO', 'Unknown']

cls = len(classes)

flux = []
CLASS = []

batch = 100
wavelengths = 3800

for idx, file in enumerate(files):
    with fits.open(file) as hdulist:
        flx = hdulist[0].data[0]
        flx = flx[:wavelengths]
        CLS = hdulist[0].header['CLASS']
    flux.append(flx)
    CLASS.append([0]*cls)
    CLASS[-1][classes.index(CLS)] = 1

'''
for i in range(cls):
    print(classes[i], ': ', sp.sum([x[i] for x in CLASS]))
'''

split = 3000

x_train = sp.array(flux[:split])
x_test = sp.array(flux[split:])

y_train = sp.array(CLASS[:split])
y_test = sp.array(CLASS[split:])

print([len(x) for x in x_train])
print(y_train)

'''need better train test split (sklearn?) and batching'''

x = tf.placeholder(tf.float32, shape = [None, wavelengths])
y_ = tf.placeholder(tf.float32, shape = [None, cls])

W = tf.Variable(tf.zeros([wavelengths,cls]))
b = tf.Variable(tf.zeros([cls]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

print('a')

for _ in range(10):
    batch = sp.random.choice(len(x_train), size = batch)
    batch_x = x_train[batch]
    batch_y = y_train[batch]
    sess.run(train_step, feed_dict={x: x_train, y_: y_train})
    '''change batching process'''

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
'''change to use test data'''
