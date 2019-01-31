import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *

class QNN(object):
    DEFUALT_HYPER = {
        'learning_rate': 0.05,
        'gamma': 0.5
    }
    def __init__(self, sess, target_state, trunc=25, n_layers=2, hyperparams=DEFUALT_HYPER):
        self.sess = sess
        self.hyperparams = hyperparams
        self.eng, self.q = sf.Engine(2)

        # Multiplier for trace penalty
        self.gamma = tf.placeholder(dtype=tf.float32, shape=[], name="gamma")
        # Learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

        self.b_splitters = tf.Variable(initial_value=tf.random_uniform([n_layers, 2], maxval=2*np.pi),
            dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi))
        self.rs = tf.Variable(initial_value=tf.random_uniform([n_layers, 2], minval=-1.4, maxval=1.4),
            dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4))
        self.alphas = tf.Variable(initial_value=tf.random_normal([n_layers, 2], mean=1, stddev=4),
            dtype=tf.float32)

        with self.eng:
            Fock(1) | self.q[0]
            Vac | self.q[1]
            for n in range(n_layers):
                self.__create_layer(n)
            MeasureFock(select=0) | self.q[0]

        self.state = self.eng.run('tf', cutoff_dim=trunc, eval=False)
        state_dm = self.state.reduced_dm(1)
        target_dm = target_state @ target_state.conj().T

        # Calculate the fidelity of the output state with the cubic phase state
        self.fid = tf.abs(tf.trace(state_dm @ target_dm)) # Output of tf.trace should be real, but usually has very small imaginary component
        norm = tf.abs(tf.trace(state_dm))
        penalty = tf.pow(norm - 1, 2) # Penalise unnormalised states
        loss = -self.fid - self.gamma*penalty

        optimiser = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.min_op = optimiser.minimize(loss)

    def train(self, epochs):
        feed_dict = {
            self.learning_rate: self.hyperparams['learning_rate'],
            self.gamma: self.hyperparams['gamma']
        }
        for step in range(epochs):
            fid_val, _ = self.sess.run([self.fid, self.min_op],
                feed_dict=feed_dict)
            print("{}: fid = {}".format(step, fid_val))

    def get_parameters(self):
        bs, r_vals, a_vals = self.sess.run([self.b_splitters, self.rs, self.alphas])
        return { 'beam_splitters': bs, 'squeezing': r_vals, 'displacement': a_vals }

    def __create_layer(self, n):
        """
        Implements one layer of a QNN
        :param q: The strawberryfields quantum register
        :param n: The layer number
        """
        # Interferometer
        BSgate(self.b_splitters[n][0], -np.pi/4) | (self.q[0], self.q[1])

        # Squeezingi
        Sgate(self.rs[n][0]) | self.q[0]
        Sgate(self.rs[n][1]) | self.q[1]

        # Interferometer
        BSgate(self.b_splitters[n][1], -np.pi/4) | (self.q[0], self.q[1])

        # Displacement
        Dgate(self.alphas[n][0]) | self.q[0]
        Dgate(self.alphas[n][1]) | self.q[1]
