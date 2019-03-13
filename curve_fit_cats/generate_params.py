# Generates and saves a set of network parameters that can be loaded in by run_curve_fit.py
import os
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *

if len(os.sys.argv) < 2:
    print("Please supply a file containing training data")
    os.sys.exit(1)

n_layers = 6
batch_size = 50
truncation = 10
gamma = 10
should_save = True
train_file = os.sys.argv[1] # File to load training data from
post_select = 0 # Photon number for post-selection measurement on ancilla mode
loss_threshold = 1.0 # Keep randomly generating params until loss < threshold

eng, q = sf.Engine(2)

# Beam splitter parameters - 2 per layer (2 interferometers)
b_splitters = tf.Variable(initial_value=tf.random_uniform([n_layers, 4], maxval=2*np.pi),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi), name='b_splitters')
# Squeezing parameters - 1 per layer
rs = tf.Variable(initial_value=tf.random_uniform([n_layers], minval=-1.4, maxval=1.4),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4), name='rs')
# Displacement parameters - 1 per layer
alphas = tf.Variable(initial_value=tf.random_normal([n_layers], mean=0, stddev=4),
    dtype=tf.float32, name='alphas')
# Cat state to input as ancilla - 1 per layer
cats = tf.Variable(initial_value=tf.random_uniform([n_layers], minval=0.0, maxval=1.0),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0), name='cats')

x = tf.placeholder(tf.float32, shape=[batch_size])
y_ = tf.placeholder(tf.float32, shape=[batch_size])

def build_layer(n):
    """
    Build one layer of the neural network
    :param n: Integer indicating which layer we are building
    """
    # Ancilla state
    Catstate(cats[n]) | q[0]

    # Interferometer
    BSgate(b_splitters[n][0], b_splitters[n][2]) | (q[0], q[1])

    # Squeezing
    Sgate(rs[n]) | q[1]

    # Interferometer
    BSgate(b_splitters[n][1], b_splitters[n][3]) | (q[0], q[1])

    # Displacement
    Dgate(alphas[n]) | q[1]

    # Measure ancilla mode
    MeasureFock(select=post_select) | q[0]

with eng:
    Dgate(x) | q[1] # Encode data as displacement along real axis

    for n in range(n_layers):
        build_layer(n)

state = eng.run('tf', cutoff_dim=truncation, eval=False, batch_size=batch_size)
output = state.quad_expectation(1)[0] # Position quadrature on mode 1
# Mean squared error
mse = tf.reduce_mean(tf.squared_difference(output, y_))
penalty = tf.squared_difference(tf.real(state.trace()), 1)
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

f = np.load(train_file)
inputs = f['x']
actual_output = f['y']
batched_data = batch_generator([inputs, actual_output], batch_size)
n_batches = inputs.size // batch_size

sess = tf.Session()
sess.run(tf.global_variables_initializer())

while True:
    total_loss = 0.0 # Keep track of cumulative loss over all batches
    for b in range(n_batches):
        batch_in, batch_out = next(batched_data) # Get next batch
        # Run a training step
        loss_val = sess.run(loss, feed_dict={
            x: batch_in,
            y_: batch_out
        })
        total_loss += loss_val
    total_loss /= n_batches # Average loss over all input data
    print("Loss: {}".format(total_loss))
    if (not np.isnan(total_loss)) and total_loss < loss_threshold:
        break
    # Re-generate random parameters
    sess.run(tf.global_variables_initializer())

if should_save:
    train_file_name = os.path.split(train_file)[-1]
    train_file_base = train_file_name.split('.')[0]
    dir_name = os.path.join('.', 'params', train_file_base)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # Save tensorflow model
    saver = tf.train.Saver(var_list=[b_splitters, rs, alphas])
    saver.save(sess, os.path.join(dir_name, 'model.ckpt'))

    # Save hyperparams, losses, and training rates using numpy
    hyperparams = {
        'n_layers': n_layers,
        'batch_size': batch_size,
        'truncation': truncation,
        'gamma': gamma,
        'post_select': post_select
    }
    # Save hyperparams to text file
    with open(os.path.join(dir_name, 'hyperparams.txt'), 'w') as h_file:
        print(hyperparams, file=h_file)
        print("Loss: {}".format(total_loss), file=h_file)

    print("Saved to: " + dir_name)
