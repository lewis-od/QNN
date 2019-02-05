import os
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from datetime import datetime
from strawberryfields.ops import *

n_layers = 3
batch_size = 50
init_learning_rate = 0.01
epochs = 50
truncation = 10
gamma = 10
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

state = eng.run('tf', cutoff_dim=truncation, eval=False, batch_size=batch_size)
output = state.quad_expectation(1)[0] # Position quadrature on mode 1
# Mean squared error
mse = tf.reduce_mean(tf.squared_difference(output, y_))
penalty = (tf.real(state.trace()) - 1) ** 2
penalty = gamma * tf.reduce_mean(penalty)
loss = mse + penalty

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
learning_rate = tf.train.exponential_decay(init_learning_rate,
    global_step, n_batches*10, 0.9, staircase=True)
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
min_op = optimiser.minimize(loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load session if command line argument given
if len(os.sys.argv) == 2:
    saver = tf.train.Saver()
    saver.restore(sess, os.sys.argv[1])
    print("Loaded saved model from " + os.sys.argv[1])

losses = np.zeros(epochs)
learning_rates = np.zeros(epochs)
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
    losses[step] = total_loss # Save loss for later
    print("{}: loss = {}".format(step, total_loss))
    lr_val = sess.run(learning_rate)
    learning_rates[step] = lr_val # Save learning rate for later

if should_save:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dir_name = os.path.join('.', 'save', now_str)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # Save tensorflow model
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(dir_name, 'model.ckpt'))

    # Save hyperparams, losses, and training rates using numpy
    hyperparams = {
        'layers': n_layers,
        'batch_size': batch_size,
        'epochs': epochs,
        'truncation': truncation,
        'gamma': gamma
    }
    output_file = os.path.join(dir_name, 'output.npz')
    if os.path.isfile(output_file):
        # Check file doesn't already exist (it will do if we loaded in a model earlier)
        output_file = os.path.join(dir_name, 'output {}.npz'.format(now_str))
    np.savez(output_file, hyperparams=hyperparams,
        loss=losses, learning_rate=learning_rates)

    # Save hyperparams to text file
    with open(os.path.join(dir_name, 'hyperparams.txt'), 'w') as f:
        print(hyperparams, file=f)

    print("Saved to " + dir_name)

import matplotlib.pyplot as plt

sparse_in = np.linspace(-2, 2, batch_size)
predicted_output = sess.run(output, feed_dict={ x: sparse_in })

input_plot = np.linspace(-2, 2, 100)
output_plot = input_plot**2

plt.subplot(1, 3, 1)
plt.plot(input_plot, output_plot)
plt.scatter(sparse_in, predicted_output, c='r', marker='x')

plt.subplot(1, 3, 2)
plt.plot(np.arange(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 3)
plt.plot(np.arange(epochs), learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.show()
