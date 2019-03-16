import os
import ast
import numpy as np
import tensorflow as tf

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

b_vals, r_vals, a_vals, c_vals = sess.run([b_splitters, rs, alphas, cats])

print("Parameter Values")
print('-'*50)
for n in range(n_layers):
    print("Layer {}".format(n))
    print("\tBS1: theta = {}, phi = {}".format(b_vals[n][0], b_vals[n][2]))
    print("\tz: {}".format(r_vals[n]))
    print("\tBS2: theta = {}, phi = {}".format(b_vals[n][1], b_vals[n][3]))
    print("\talpha: {}".format(a_vals[n]))
    print("\tcat: {}".format(c_vals[n]))
    print('-'*50)


