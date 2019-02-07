import strawberryfields as sf
import tensorflow as tf
import numpy as np
from qnn.base import QNNBase
from strawberryfields.ops import Dgate, MeasureX

class CurveFitter(QNNBase):
    DEFAULT_HYPER_CF = {
        'gamma': 10
    }
    def __init__(self, sess, batch_size, n_modes=2, n_layers=6, hyperparams={}):
        for key, val in self.DEFAULT_HYPER_CF.items():
            if key not in hyperparams: hyperparams[key] = val
        super(CurveFitter, self).__init__(
            sess, batch_size, n_modes, n_layers, hyperparams
        )

    def build_encoder(self):
        # Encode data by displacing along real axis
        Dgate(self.x) | self.q[1]

    def loss_fn(self):
        state = self.eng.run('tf', cutoff_dim=self.hyperparams['trunc'],
            batch_size=self.batch_size, eval=False)
        exp_val = state.quad_expectation(1)[0]
        # Mean squared error
        mse = tf.reduce_mean(tf.squared_difference(exp_val, self.y_))
        # Trace penalty
        penalty = (tf.real(state.trace()) - 1)**2 # TODO: Replace with squared_difference
        penalty = self.hyperparams['gamma'] * tf.reduce_mean(penalty)
        loss = mse + penalty
        return loss

    def make_prediction(self, input):
        """Runs the circuit with given input values"""
        if input.size > self.batch_size:
            # TODO: Allow input of any size
            raise NotImplementedError("Input size must be <= batch size (for now)")

        # Pad inputs with zeros
        inputs = np.zeros(self.batch_size)
        inputs[:input.size] = input

        # Evaluate outputs
        states = self.eng.run('tf', trunc=self.hyperparams['trunc'],
            batch_size=self.batch_size, eval=False)
        outputs = states.quad_expectation(1)[0]
        outputs = self.sess.run(outputs, feed_dict={ self.x: inputs })
        outputs = outputs[:input.size]

        return outputs
