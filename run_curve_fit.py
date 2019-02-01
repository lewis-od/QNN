import os
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from datetime import datetime
from strawberryfields.ops import *

n_layers = 3
batch_size = 5
init_learning_rate = 0.005
epochs = 20
should_save = True

eng, q = sf.Engine(2)

b_splitters = tf.Variable(initial_value=tf.random_uniform([n_layers, 2], maxval=2*np.pi),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi))
# Squeezing parameters - 2 per layer (1 on each mode)
rs = tf.Variable(initial_value=tf.random_uniform([n_layers, 2], minval=-1.4, maxval=1.4),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4))
# Displacement parameters - 2 per layer (1 on each mode)
alphas = tf.Variable(initial_value=tf.random_normal([n_layers, 2], mean=1, stddev=4),
    dtype=tf.float32)

x = tf.placeholder(tf.float32, shape=[batch_size])
y_ = tf.placeholder(tf.float32, shape=[batch_size])

def build_layer(n):
    # Ancilla state
    Fock(1) | q[0]

    # Interferometer
    BSgate(b_splitters[n][0], -np.pi/4) | (q[0], q[1])

    # Squeezing
    Sgate(rs[n][0]) | q[0]
    Sgate(rs[n][1]) | q[1]

    # Interferometer
    BSgate(b_splitters[n][1], -np.pi/4) | (q[0], q[1])

    # Displacement
    Dgate(alphas[n][0]) | q[0]
    Dgate(alphas[n][1]) | q[1]

    # Measure ancilla mode
    MeasureFock(select=0) | q[0]

with eng:
    Dgate(x) | q[1]

    for n in range(n_layers):
        build_layer(n)

state = eng.run('tf', cutoff_dim=20, eval=False, batch_size=batch_size)
output = state.quad_expectation(1)[0] # Position quadrature on mode 1
# Mean squared error
loss = tf.reduce_mean(tf.squared_difference(output, y_))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 5, 0.9)
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
min_op = optimiser.minimize(loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input = np.linspace(-1, 1, batch_size)
actual_output = input ** 2 # Curve fit f(x) = x**2

for step in range(epochs):
    loss_val, _ = sess.run([loss, min_op], feed_dict={ x: input, y_: actual_output })
    print("{}: loss = {}".format(step, loss_val))
    if step % 5 == 0:
        lr_val = sess.run(learning_rate)
        print("Learning rate: {}".format(lr_val))

if should_save:
    saver = tf.train.Saver()
    dir_name = './save/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    saver.save(sess, dir_name + 'model.ckpt')
    print("Saved to " + dir_name)

# saver = tf.train.Saver()
# saver.restore(sess, './save/2019-02-01 17:42:47/model.ckpt')
#
# import matplotlib.pyplot as plt
#
# predicted_output = sess.run(output, feed_dict={ x: input })
# input_plot = np.linspace(-1, 1, 50)
# output_plot = input_plot**2
#
# plt.plot(input_plot, output_plot)
# plt.scatter(input, predicted_output, c='r', marker='x')
# plt.show()
