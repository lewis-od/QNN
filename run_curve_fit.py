import os
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from datetime import datetime
from strawberryfields.ops import *

n_layers = 3
batch_size = 25
init_learning_rate = 0.005
epochs = 50
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
    Dgate(x) | q[1] # Encode data as displacement along real axis

    for n in range(n_layers):
        build_layer(n)

state = eng.run('tf', cutoff_dim=20, eval=False, batch_size=batch_size)
output = state.quad_expectation(1)[0] # Position quadrature on mode 1
# Mean squared error
loss = tf.reduce_mean(tf.squared_difference(output, y_))

def batch_generator(arrays, b_size):
    """Groups data in the arrays list into batches of size b_size"""
    starts = [0] * len(arrays)
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + b_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += b_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
        yield batches

inputs = (np.random.rand(100) - 0.5) * 4 # Random values in [-2,2)
actual_output = inputs ** 2 # Curve fit f(x) = x**2
batched_data = batch_generator([inputs, actual_output], batch_size)
n_batches = inputs.size // batch_size

# Train using gradient descent
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, n_batches*10, 0.9)
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
min_op = optimiser.minimize(loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load session if command line argument given
if len(os.sys.argv) == 2:
    saver = tf.train.Saver()
    saver.restore(sess, os.sys.argv[1])
    print("Loaded saved model from " + os.sys.argv[1])

for step in range(epochs):
    total_loss = 0.0 # Keep track of cumulative loss over all batches
    for b in range(n_batches):
        batch_in, batch_out = next(batched_data) # Get next batch
        # Run a training step
        loss_val, _ = sess.run([loss, min_op], feed_dict={
            x: batch_in,
            y_: batch_out
        })
        total_loss += loss_val
    total_loss /= n_batches # Average loss over all input data
    print("{}: loss = {}".format(step, total_loss))
    if step % 5 == 0: # Print the learning rate every 5 steps
        lr_val = sess.run(learning_rate)
        print("Learning rate: {}".format(lr_val))

if should_save:
    # Save the trained network
    saver = tf.train.Saver()
    dir_name = './save/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    saver.save(sess, dir_name + 'model.ckpt')
    print("Saved to " + dir_name)

import matplotlib.pyplot as plt

sparse_in = np.linspace(-2, 2, batch_size)
predicted_output = sess.run(output, feed_dict={ x: sparse_in })

input_plot = np.linspace(-2, 2, 100)
output_plot = input_plot**2

plt.plot(input_plot, output_plot)
plt.scatter(sparse_in, predicted_output, c='r', marker='x')
plt.show()
