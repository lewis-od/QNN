import os
import ast
import numpy as np
import tensorflow as tf
import strawberryfields as sf
import matplotlib.pyplot as plt
from datetime import datetime
from strawberryfields.ops import *
from qutip.matplotlib_utilities import wigner_cmap

# ----- Parse command-line args -----

if len(os.sys.argv) < 2:
    print("Please provide a folder containing results.")
    os.sys.exit(1)

res_folder = os.sys.argv[1]

truncation = 35 # Cutoff dimension for strawberry fields
x_in = 1.0

# ----- Hyperparameters -----

n_layers = 6 # Number of layers in neural network
ancilla_state_n = 0 # Photon number of ancilla Fock state
post_select = 0 # Photon number for post-selection measurement on ancilla mode
train_file = '' # File to load training data from

# ----- Load in hyperparameters -----

hyper_loc = os.path.join(res_folder, 'hyperparams.txt')
with open(hyper_loc, 'r') as hyper_file:
    hyper_str = hyper_file.readline() # Hyperparam dict on first line
    hyper_dict = ast.literal_eval(hyper_str) # Convert str->dict

    dont_want = ['epochs', 'gamma', 'truncation', 'batch_size']
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

x = tf.placeholder(tf.float32, shape=[]) # Input to neural network

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
    Fock(ancilla_state_n) | q[0]

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

state = eng.run('tf', cutoff_dim=truncation, session=sess, feed_dict={ x: x_in })

xvec = np.linspace(-5, 5, 150)
wigner = state.wigner(1, xvec, xvec)
w_map = wigner_cmap(wigner)

levels = np.linspace(wigner.min(), wigner.max(), 35)

plt.contourf(xvec, xvec, wigner, levels=levels, cmap=w_map)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('p')
plt.title("x_in = {}, T = {}".format(x_in, truncation))
plt.show()
