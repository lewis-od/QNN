import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *

# Parameters for the circuit
b_splitters = tf.Variable(initial_value=tf.random_uniform([2, 2], maxval=2*np.pi),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi))
rs = tf.Variable(initial_value=tf.random_uniform([2], minval=-1.4, maxval=1.4),
    dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4))
alphas = tf.Variable(initial_value=tf.random_normal([2], mean=1, stddev=4),
    dtype=tf.float32)

eng, q = sf.Engine(2)
with eng:
    Fock(1) | q[0]
    Vac | q[1]

    # Interferometer
    BSgate(b_splitters[0][0], b_splitters[0][1]) | (q[0], q[1])

    # Squeezing
    Sgate(rs[0]) | q[0]
    Sgate(rs[1]) | q[1]

    # Interferometer
    BSgate(b_splitters[1][0], b_splitters[1][0]) | (q[0], q[1])

    # Displacement
    Dgate(alphas[0]) | q[0]
    Dgate(alphas[1]) | q[1]

    # Measurement
    MeasureFock(select=0) | q[0]

# Run the circuit
state = eng.run('tf', cutoff_dim=25, eval=False)
# Trace out the ancilla mode
state_dm = state.reduced_dm(1)

# TODO: Change this to something useful
prob = state.fock_prob([1, 1])
loss = -prob

optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.01)
min_op = optimiser.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10):
    if step == 0:
        bs, r_vals, a_vals = sess.run([b_splitters, rs, alphas])
        print(bs)
        print(r_vals)
        print(a_vals)
    prob_val, _ = sess.run([prob, min_op])
    print("{}: prob = {}".format(step, prob_val))
