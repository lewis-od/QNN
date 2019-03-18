import os
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from datetime import datetime
from strawberryfields.ops import *

# ----- Hyperparameters -----

n_layers = 6 # Number of layers in neural network
batch_size = 50 # Batch size used in training
epochs = 2 # Number of epochs to use
truncation = 10 # Cutoff dimension for strawberry fields
gamma = 10 # Multiplier for trace penalty
should_save = True # Whether or not to save the results
post_select = 5 # Photon number for post-selection measurement on ancilla mode
train_file = 'sinc.npz' # File to load training data from

# ----- Tensorflow variables -----

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

sess = tf.Session()
# Load in network parameters that were saved by generate_params.py
dataset_name = train_file.split('.')[0]
ckpt_file = os.path.join('params', dataset_name, 'model.ckpt')
# Load in saved parameters
try:
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)
    print("Loaded model from " + ckpt_file)
except:
    print("Unable to load model from " + ckpt_file)
    sess.run(tf.global_variables_initializer())

x = tf.placeholder(tf.float32, shape=[batch_size]) # Input to neural network
y_ = tf.placeholder(tf.float32, shape=[batch_size]) # Expected output (used to calculate loss)

# ----- Build neural network circuit using strawberryfields -----

eng, q = sf.Engine(2) # Intitialise strawberry fields with a 2 mode system

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

# Build the neural network
with eng:
    Dgate(x) | q[1] # Encode input data as displacement along real axis

    for n in range(n_layers):
        build_layer(n)

# ----- Run the simulation and calculate the loss function -----

state = eng.run('tf', cutoff_dim=truncation, eval=False, batch_size=batch_size)
output = state.quad_expectation(1)[0] # Position quadrature on mode 1
# Mean squared error
mse = tf.reduce_mean(tf.squared_difference(output, y_))
# Add trace penalty
penalty = tf.squared_difference(tf.real(state.trace()), 1)
penalty = gamma * tf.reduce_mean(penalty)
loss = mse + penalty

# ----- Load and prepare training data -----

with np.load(os.path.join('training', train_file)) as f:
    # Create Tensorflow Dataset from training data
    dataset = tf.data.Dataset.from_tensors((f['x'], f['y']))
    # Group into batches of batch_size
    batched_data = dataset.repeat().take(batch_size).make_one_shot_iterator()
    # Used in plotting later
    x_true = f['x_true']
    y_true = f['y_true']
dataset_size = dataset.output_shapes[0].as_list()[0]
n_batches = dataset_size // batch_size

# ----- Train the neural network -----

global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(0.05, global_step, n_batches*5, 0.95)
# optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
# In original paper describing AdaDelta, no learning rate parameter is required.
# Setting learning_rate = 1.0 in the tensorflow implementation of the algorithm
# mimics this.
optimiser = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95)
min_op = optimiser.minimize(loss, global_step=global_step)

# Can't run tf.global_variables_initializer() as this will overwrite parameter
# values that may have been loaded in
init_op = tf.variables_initializer(optimiser.variables() + [global_step])
sess.run(init_op)

start_time = datetime.now()

losses = np.zeros(epochs) # Array to keep track of loss after each epoch
for step in range(epochs):
    total_loss = 0.0 # Keep track of cumulative loss over all batches
    for b in range(n_batches):
        batch_in, batch_out = sess.run(batched_data.get_next()) # Get next batch
        # Run a training step
        loss_val, _ = sess.run([loss, min_op], feed_dict={
            x: batch_in,
            y_: batch_out
        })
        total_loss += loss_val
    total_loss /= n_batches # Average loss over all batches
    print("[{}]: {}".format(step, total_loss))
    if np.isnan(total_loss): # Exit if loss has diverged
        os.sys.exit(1)
    losses[step] = total_loss # Save loss value for this epoch

time_elapsed = datetime.now() - start_time
print("Time Elapsed: {}".format(time_elapsed))

# ----- Save and plot results -----

# Get entire training set from batched Tensorflow Dataset
train_in, train_out = sess.run(dataset.make_one_shot_iterator().get_next())

# Calculate the predictions of the network
sparse_in = np.linspace(train_in.min(), train_in.max(), batch_size)
predicted_output = sess.run(output, feed_dict={ x: sparse_in })

if should_save:
    # Create a folder to save results
    training_set = train_file.split('.')[0]
    dir_str = '{}-{}'.format(training_set, post_select)
    dir_name = os.path.join('.', 'results', dir_str)
    if os.path.isdir(dir_name): # Add timestamp if folder already exists
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dir_name += now_str
    os.makedirs(dir_name)

    # Save tensorflow model
    saver = tf.train.Saver(var_list=[b_splitters, rs, alphas, cats])
    saver.save(sess, os.path.join(dir_name, 'model.ckpt'))

    # Save hyperparams, losses, and training rates using numpy
    hyperparams = {
        'n_layers': n_layers,
        'batch_size': batch_size,
        'epochs': epochs,
        'truncation': truncation,
        'gamma': gamma,
        'post_select': post_select
    }
    output_file = os.path.join(dir_name, 'output.npz')
    # Save hyperparams and loss values
    np.savez(output_file, hyperparams=hyperparams, loss=losses,
        inputs=sparse_in, predictions=predicted_output)

    # Save hyperparams to text file
    with open(os.path.join(dir_name, 'hyperparams.txt'), 'w') as h_file:
        print(hyperparams, file=h_file)
        print("Training data: " + train_file, file=h_file)
        print("Optimiser: " + optimiser.get_name(), file=h_file)
        print("Loss: {}".format(losses[-1]), file=h_file)

    print("Saved to: " + dir_name)

import matplotlib.pyplot as plt

# Plot predictions, training data, and "true" (non-noisy) curve
plt.subplot(1, 2, 1)
plt.plot(x_true, y_true, c='g')
plt.scatter(sparse_in, predicted_output, c='r', marker='x')
plt.scatter(train_in, train_out, c='b', marker='o')

# Plot loss values
plt.subplot(1, 2, 2)
plt.plot(np.arange(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()
