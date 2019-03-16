import os
import ast
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from datetime import datetime
from strawberryfields.ops import *

# ----- Parse command-line args -----

if len(os.sys.argv) < 2:
    print("Please provide a folder containing results.")
    os.sys.exit(1)

res_folder = os.sys.argv[1]

# ----- Hyperparameters -----

n_layers = 6 # Number of layers in neural network
batch_size = 50 # Batch size used in training
truncation = 10 # Cutoff dimension for strawberry fields
post_select = 0 # Photon number for post-selection measurement on ancilla mode
train_file = '' # File to load training data from

# ----- Load in hyperparameters -----

hyper_loc = os.path.join(res_folder, 'hyperparams.txt')
with open(hyper_loc, 'r') as hyper_file:
    hyper_str = hyper_file.readline() # Hyperparam dict on first line
    hyper_dict = ast.literal_eval(hyper_str) # Convert str->dict

    dont_want = ['epochs', 'gamma']
    for p_name, p_val in hyper_dict.items(): # Set required hyperparams
        if p_name in dont_want: continue
        globals()[p_name] = p_val # Probably dangerous
        print("{} = {}".format(p_name, eval(p_name)))

    train_file_name = hyper_file.readline().split(': ')[-1]
    train_file_name = train_file_name.strip()
    train_file = os.path.join(os.path.pardir, 'training', train_file_name)

    print("Loaded hyperparams.txt")

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

x = tf.placeholder(tf.float32, shape=[batch_size]) # Input to neural network
y_ = tf.placeholder(tf.float32, shape=[batch_size]) # Expected output (used to calculate loss)

# ----- Load in network parmeters -----

sess = tf.Session()
# Load in network parameters that were saved by generate_params.py
dataset_name = train_file.split('.')[0]
ckpt_file = os.path.join(res_folder, 'model.ckpt')
# Load in saved parameters
try:
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)
    print("Loaded model from " + ckpt_file)
except:
    print("Unable to load model from " + ckpt_file)
    os.sys.exit(1)

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

# ----- Setup Tensorflow graph for running simulation -----

state = eng.run('tf', cutoff_dim=truncation, eval=False, batch_size=batch_size)
output = state.quad_expectation(1)[0] # Position quadrature on mode 1

# ----- Load and prepare training data -----

# Load in training data
f = np.load(train_file)
inputs = f['x']
actual_output = f['y']

# ----- Run data through network -----

# Calculate the predictions of the network
sparse_in = np.linspace(inputs.min(), inputs.max(), batch_size)
print("Running simulation")
predicted_output = sess.run(output, feed_dict={ x: sparse_in })

# ----- Plot results -----

print("Plotting results")
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot predictions, training data, and "true" (non-noisy) curve
true_curve, = plt.plot(f['x_true'], f['y_true'], c='g')
predicted_scatter = plt.scatter(sparse_in, predicted_output, c='r', marker='x')
input_scatter = plt.scatter(inputs, actual_output, c='b', marker='o')
plt.legend([true_curve, input_scatter, predicted_scatter],
    ['True Curve', 'Training Data', 'Prediction'], scatterpoints=1)
plt.xlabel("$x$", fontsize=18)

y_str = r'$f(x)$'
if 'sinc' in train_file:
    y_str = r'sinc$(x)$'
elif 'sin' in train_file:
    y_str = r'$\sin (x)$'
elif 'x_cubed' in train_file:
    y_str = r'$x^3$'

plt.ylabel(y_str, fontsize=18)

save_path = os.path.join(res_folder, 'plot.eps')
plt.savefig(save_path, bbox_inches='tight')
print("Saved plot to: " + save_path)

plt.show()
