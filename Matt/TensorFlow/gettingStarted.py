import tensorflow as tf
import numpy as np

'''
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
x_train = [1,2,3,4]
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
'''

feature_columns = [tf.feature_column.numeric_column('x', shape = [1])]

estimator = tf.estimator.LinearRegressor(feature_columns = feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size = 4, num_epochs=None, shuffle = True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size = 4, num_epochs = 1000, shuffle = False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, y_eval, batch_size = 4, num_epochs = 1000, shuffle = False)

estimator.train(input_fn = input_fn, steps = 1000)

train_metrics = estimator.evaluate(input_fn = train_input_fn)
eval_metrics = estimator.evaluate(input_fn = eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
