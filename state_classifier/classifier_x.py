import os
import numpy as np
import strawberryfields as sf
import tensorflow as tf
from datetime import datetime
from strawberryfields.ops import *

# ----- Hyperparameters ------
n_layers = 6
batch_size = 100
epochs = 10
truncation = 10
gamma = 10

# Encode outputs as locations in phase space
gaussian_x = 1.0
gaussian_p = 0.0
non_gaussian_x = -1.0
non_gaussian_p = 0.0

# ----- Setup tensorflow variables -----

eng, q = sf.Engine(1)

b_splitters = tf.Variable(initial_value=tf.random_uniform([n_layers, 2], maxval=2*np.pi),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi))
# Squeezing parameters - 2 per layer (1 on each mode)
rs = tf.Variable(initial_value=tf.random_uniform([n_layers], minval=-1.4, maxval=1.4),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4))
# Displacement parameters - 2 per layer (1 on each mode)
alphas = tf.Variable(initial_value=tf.random_normal([n_layers], mean=0, stddev=4),
    dtype=tf.float32)
# Kerr gate parameters
kappas = tf.Variable(initial_value=tf.random_normal([n_layers], mean=0, stddev=10),
    dtype=tf.float32)

x = tf.placeholder(tf.complex64, shape=[batch_size, 3]) # Parameters for input states
y_ = tf.placeholder(tf.int32, shape=[2, batch_size]) # Labels in logit format

# ----- Build quantum circuit -----

def build_layer(n):
    # Interferometer
    Rgate(b_splitters[n][0]) | q[0]

    # Squeezing
    Sgate(rs[n]) | q[0]

    # Interferometer
    Rgate(b_splitters[n][1]) | q[0]

    # Displacement
    Dgate(alphas[n]) | q[0]

    # Kerr gate
    Kgate(kappas[n]) | q[0]

with eng:
    # Create input state
    Sgate(x[:, 0]) | q[0]
    Dgate(x[:, 1]) | q[0]
    Kgate(x[:, 2]) | q[0]

    for n in range(n_layers):
        build_layer(n)

# ------ Load and prepare training data -----

f = np.load('data/states.npz')
in_states = f['states']
indices = f['labels'].astype(np.int) # List containing 0s or 1s
# Permute data to mix gaussian <-> non-gaussian
perm = np.random.permutation(indices.size)
in_states = in_states[perm, :]
indices = indices[perm]

# ----- Run circuit and calculate loss -----

state = eng.run('tf', cutoff_dim=truncation, eval=False, batch_size=batch_size)
output_x = state.quad_expectation(0)[0]
output_p = state.quad_expectation(0, phi=np.pi/2.0)[0]

# Distance to point in phase space corresponding to input being Gaussian
dx_g = tf.squared_difference(output_x, gaussian_x)
dp_g = tf.squared_difference(output_p, gaussian_p)
dist_g = tf.sqrt(dx_g + dp_g)

# Distance to point in phase space corresponding to input being non-Gaussian
dx_ng = tf.squared_difference(output_x, non_gaussian_x)
dp_ng = tf.squared_difference(output_p, non_gaussian_p)
dist_ng = tf.sqrt(dx_ng + dp_ng)

# Smaller distance => Larger probability
act_g = tf.reciprocal(dist_g)
act_ng = tf.reciprocal(dist_ng)

# If dist = 0, act will be inf. Replace with suitably large number instead
act_g = tf.where(tf.is_inf(act_g), tf.ones_like(act_g)*1e5, act_g)
act_ng = tf.where(tf.is_inf(act_ng), tf.ones_like(act_ng)*1e5, act_ng)

logits = tf.stack([act_g, act_ng])
softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
softmax_loss = tf.reduce_mean(softmax_loss)

penalty = tf.squared_difference(tf.real(state.trace()), 1)
penalty = gamma * tf.reduce_mean(penalty)
loss = softmax_loss + penalty

# ----- Train the network -----

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

batched_data = batch_generator([in_states, indices], batch_size)
n_batches = indices.size // batch_size

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.1, global_step, n_batches*10, 0.95)
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
min_op = optimiser.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses = np.zeros(epochs)
batch_labels = np.zeros([2, batch_size]) # List containing [1; 0] or [0; 1]
for step in range(epochs):
    total_loss = 0.0
    for b in range(n_batches):
        batch_in, batch_indices = next(batched_data)
        for i in range(2): # Convert 0 -> [1; 0] and 1 -> [0; 1]
            batch_labels[i, np.where(batch_indices == i)] = 1

        loss_val, _ = sess.run([loss, min_op], feed_dict={
            x: batch_in,
            y_: batch_labels
        })
        total_loss += loss_val
    total_loss /= n_batches
    losses[step] = total_loss
    print("[{}]: {}".format(step, total_loss))
    if np.isnan(total_loss):
        os.sys.exit(1)

# ----- Save model -----

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
dir_name = os.path.join('.', 'save', now_str)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

# Save tensorflow model
saver = tf.train.Saver(var_list=[b_splitters, rs, alphas, kappas])
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

# ----- Plot results -----

# Run selection of states through the network to plot the outputs
gaussian_params = in_states[indices == 0, :]
gaussian_params = gaussian_params[:batch_size, :]
non_gaussian_params = in_states[indices == 1, :]
non_gaussian_params = non_gaussian_params[:batch_size, :]

gaussian_x, gaussian_p = sess.run([output_x, output_p], feed_dict={ x: gaussian_params })
non_gaussian_x, non_gaussian_p = sess.run([output_x, output_p], feed_dict={ x: non_gaussian_params })

import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.figure()
plt.scatter(gaussian_x, gaussian_p, marker='x', color='b')
plt.scatter(non_gaussian_x, non_gaussian_p, marker='x', color='r')
plt.xlabel("x")
plt.ylabel("p")

plt.show()
