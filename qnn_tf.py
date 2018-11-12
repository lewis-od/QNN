import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
from states import cubic_phase

# Truncation to use in simulation
trunc = 25

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
state = eng.run('tf', cutoff_dim=trunc, eval=False)
# Trace out the ancilla mode
state_dm = state.reduced_dm(1)

# Calculate density matrix of an approximate cubic phase state
target_state = cubic_phase(trunc, 0.005, -1.0).full()
target_dm = target_state @ target_state.conj().T

# Calculate the fidelity of the output state with the cubic phase state
fid = tf.trace(state_dm @ target_dm)
loss = -tf.abs(fid)

# Tell tensorflow what to optimise
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.01)
min_op = optimiser.minimize(loss)

# Create the tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Run 10 training steps
for step in range(10):
    if step == 0:
        bs, r_vals, a_vals = sess.run([b_splitters, rs, alphas])
        print("Initial parameters:")
        print("Beam splitters:")
        print(bs)
        print("Squeezing:")
        print(r_vals)
        print("Displacements:")
        print(a_vals)
    fid_val, _ = sess.run([fid, min_op])
    print("{}: fidelity = {}".format(step, fid_val))

# Print results
bs, r_vals, a_vals = sess.run([b_splitters, rs, alphas])
print("Final parameters:")
print("Beam splitters:")
print(bs)
print("Squeezing:")
print(r_vals)
print("Displacements:")
print(a_vals)
