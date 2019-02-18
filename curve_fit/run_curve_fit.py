import os
import ast
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from datetime import datetime
from strawberryfields.ops import *

n_layers = 6 # Number of layers in neural network
batch_size = 50 # Batch size used in training
epochs = 500 # Number of epochs to use
truncation = 10 # Cutoff dimension for strawberry fields
gamma = 10 # Multiplier for trace penalty
should_save = True # Whether or not to save the results

eng, q = sf.Engine(2) # Intitialise strawberry fields with a 2 mode system

# Beam splitter parameters - 2 per layer (2 interferometers)
b_splitters = tf.Variable(initial_value=tf.random_uniform([n_layers, 2], maxval=2*np.pi),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi), name='b_splitters')
# Squeezing parameters - 2 per layer (1 on each mode)
rs = tf.Variable(initial_value=tf.random_uniform([n_layers, 2], minval=-1.4, maxval=1.4),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4), name='rs')
# Displacement parameters - 2 per layer (1 on each mode)
alphas = tf.Variable(initial_value=tf.random_normal([n_layers, 2], mean=0, stddev=4),
    dtype=tf.float32, name='alphas')

sess = tf.Session()
# Load in network parameters that were saved by generate_params.py
if len(os.sys.argv) == 2:
    ckpt_file = os.sys.argv[1]
    # Load in saved parameters
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    # File containing hyperparameter values (n_layers, batch_size, etc)
    # If these are different from the ones set above, we will get errors
    hyper_file = os.path.join(os.path.split(ckpt_file)[0], "hyperparams.txt")
    with open(hyper_file, 'r') as f:
        param_str = f.readline() # Hyperparams dict on first line of file
    loaded_params = ast.literal_eval(param_str) # Convert string -> dict
    for param_name, val in loaded_params.items():
        try:
            actual_val = eval(param_name) # Value of hyperparam set in this file
        except Exception:
            print("Error parsing hyperparams.txt")
            os.sys.exit(1)
        if actual_val != val:
            print("Error: Loaded parameters not compatible with current settings")
            print("Expected {} to be {} but got {}".format(param_name, actual_val, val))
            os.sys.exit(1)
    print("Loaded saved model from " + ckpt_file)
else: # No file provided, generate parameters randomly
    sess.run(tf.global_variables_initializer())

x = tf.placeholder(tf.float32, shape=[batch_size]) # Input to neural network
y_ = tf.placeholder(tf.float32, shape=[batch_size]) # Expected output (used to calculate loss)

def build_layer(n):
    """
    Build one layer of the neural network
    :param n: Integer indicating which layer we are building
    """
    # Ancilla state
    Vac | q[0]

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
    MeasureFock(select=2) | q[0]

# Build the neural network
with eng:
    Dgate(x) | q[1] # Encode input data as displacement along real axis

    for n in range(n_layers):
        build_layer(n)

state = eng.run('tf', cutoff_dim=truncation, eval=False, batch_size=batch_size)
output = state.quad_expectation(1)[0] # Position quadrature on mode 1
# Mean squared error
mse = tf.reduce_mean(tf.squared_difference(output, y_))
# Add trace penalty
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

# Load in training data
f = np.load('training/sinc.npz')
inputs = f['x']
actual_output = f['noisy']
batched_data = batch_generator([inputs, actual_output], batch_size)
n_batches = inputs.size // batch_size

global_step = tf.Variable(0, trainable=False)
# In original paper describing AdaDelta, no learning rate parameter is required.
# Setting learning_rate = 1.0 in the tensorflow implementation of the algorithm
# mimics this.
optimiser = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95)
min_op = optimiser.minimize(loss, global_step=global_step)

# Can't run tf.global_variables_initializer() as this will overwrite parameter
# values that may have been loaded in
init_op = tf.variables_initializer(optimiser.variables() + [global_step])
sess.run(init_op)

losses = np.zeros(epochs) # Array to keep track of loss after each epoch
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
    total_loss /= n_batches # Average loss over all batches
    print("{}: loss = {}".format(step, total_loss))
    if np.isnan(total_loss): # Exit if loss has diverged
        os.sys.exit(1)
    losses[step] = total_loss # Save loss value for this epoch

if should_save:
    # Create a folder with the current date and time
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dir_name = os.path.join('.', 'save', now_str)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # Save tensorflow model
    saver = tf.train.Saver(var_list=[b_splitters, rs, alphas])
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
        # Check file doesn't already exist
        output_file = os.path.join(dir_name, 'output {}.npz'.format(now_str))
    # Save hyperparams and loss values
    np.savez(output_file, hyperparams=hyperparams, loss=losses)

    # Save hyperparams to text file
    with open(os.path.join(dir_name, 'hyperparams.txt'), 'w') as h_file:
        print(hyperparams, file=h_file)
        print("Optimiser: " + optimiser.get_name(), file=h_file)

    print("Saved to: " + dir_name)

import matplotlib.pyplot as plt

# Calculate the predictions of the network
sparse_in = np.linspace(-2, 2, batch_size)
predicted_output = sess.run(output, feed_dict={ x: sparse_in })

# Plot predictions, training data, and "true" (non-noisy) curve
plt.subplot(1, 2, 1)
plt.plot(inputs, f['true'], c='g')
plt.scatter(sparse_in, predicted_output, c='r', marker='x')
plt.scatter(inputs, actual_output, c='b', marker='o')

# Plot loss values
plt.subplot(1, 2, 2)
plt.plot(np.arange(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()
