import strawberryfields as sf
import tensorflow as tf
import numpy as np
from qnn.base import QNNBase
from strawberryfields.ops import *

class CurveFitter(QNNBase):
    DEFAULT_HYPER_CF = {
        'gamma': 10
    }
    def __init__(self, sess, batch_size, n_layers=6, hyperparams={}):
        for key, val in self.DEFAULT_HYPER_CF.items():
            if key not in hyperparams: hyperparams[key] = val
        super(CurveFitter, self).__init__(
            sess, batch_size, 2, n_layers, hyperparams
        )

    def build_encoder(self):
        # Encode data by displacing along real axis
        Dgate(self.x) | self.q[0]

    def loss_fn(self):
        state = self.eng.run('tf', cutoff_dim=self.hyperparams['cutoff'],
            batch_size=self.batch_size, eval=False)
        exp_val = state.quad_expectation(0)[0]
        # Mean squared error
        mse = tf.reduce_mean(tf.squared_difference(exp_val, self.y_))
        # Trace penalty
        penalty = (tf.real(state.trace()) - 1)**2 # TODO: Replace with squared_difference
        penalty = self.hyperparams['gamma'] * tf.reduce_mean(penalty)
        loss = mse + penalty
        return loss

    def make_prediction(self, inputs):
        """
        Runs the neural network with given input values

        :param inputs: A 1D numpy array of input values
        """
        outputs = np.zeros(inputs.size) # Array to hold output vals
        # Input to network must be of length self.batch_size
        # If inputs.size > self.batch_size, we will have to call the network
        # multiple times.
        evaluations = int(np.ceil(inputs.size / self.batch_size))
        # Array to hold input values for each evaluation
        batch_in = np.zeros(self.batch_size)
        for n in range(evaluations):
            start = n * self.batch_size
            end = np.min([inputs.size, (n+1)*self.batch_size])
            batch_in[:(end-start)] = inputs[start:end]
            states = self.eng.run('tf', cutoff_dim=self.hyperparams['cutoff'],
                batch_size=self.batch_size, eval=False)
            preds = states.quad_expectation(0)[0]
            preds = self.sess.run(preds, feed_dict={ self.x: batch_in })
            outputs[start:end] = preds[:(end-start)]

        return outputs

class CurveFitterX(CurveFitter):
    def __init__(self, sess, batch_size, n_layers=6, hyperparams={}):
        for key, val in self.DEFAULT_HYPER_CF.items():
            if key not in hyperparams: hyperparams[key] = val
        # Call QNNBase constructor, not CurveFitter
        super(CurveFitter, self).__init__(
            sess, batch_size, 1, n_layers, hyperparams
        )

    # Overriden from QNNBase
    def _init_params(self):
        # Phase shift params
        self.b_splitters = tf.Variable(
            initial_value=tf.random_uniform([self.n_layers, 2], maxval=2*np.pi),
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi))
        # Squeezing parameters
        self.rs = tf.Variable(
            initial_value=tf.random_uniform([self.n_layers], minval=-1.4, maxval=1.4),
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4))
        # Displacement parameters
        self.alphas = tf.Variable(
            initial_value=tf.random_normal([self.n_layers], mean=0, stddev=4),
            dtype=tf.float32)
        # Kerr gate parameters
        self.kappas = tf.Variable(
            initial_value=tf.random_normal([self.n_layers], mean=0, stddev=10),
            dtype=tf.float32)

    # Overriden from QNNBase
    def _build_layer(self, n):
        Rgate(self.b_splitters[n][0]) | self.q[0]
        Sgate(self.rs[n]) | self.q[0]
        Rgate(self.b_splitters[n][1]) | self.q[0]
        Dgate(self.alphas[n]) | self.q[0]
        Kgate(self.kappas[n]) | self.q[0]
