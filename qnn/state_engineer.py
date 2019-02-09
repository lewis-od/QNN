import tensorflow as tf
import numpy as np
from qnn.base import QNNBase
from strawberryfields.ops import Vac

class StateEngineer(QNNBase):
    DEFAULT_HYPER_SE = {
        'gamma': 10
    }
    def __init__(self, sess, target_state, n_layers=3, hyperparams={}):
        for key, val in self.DEFAULT_HYPER_SE.items():
            if key not in hyperparams.keys(): hyperparams[key] = val

        self.batch_size = 1
        self.target_state = target_state

        # Set up neural network
        super(StateEngineer, self).__init__(sess, batch_size=1, n_modes=2,
            n_layers=n_layers, hyperparams=hyperparams)

    # State engineering has a fixed input state (the vacuum)
    def build_encoder(self):
        Vac | self.q[0]

    def loss_fn(self):
        """Calculate the fidelity of the output state with self.target_state"""
        # Calculate state by simulating circuit
        state = self.eng.run('tf', cutoff_dim=self.hyperparams['cutoff'],
            batch_size=None, eval=False)
        state_dm = state.reduced_dm(0) # Trace out ancilla mode
        # Convert target_state from vector -> density matrix
        target_dm = self.target_state @ self.target_state.conj().T

        # Calculate the fidelity of the output state with the cubic phase state
        # Output of tf.trace should be real, but can have small imaginary part
        self.fid = tf.abs(tf.trace(state_dm @ target_dm))
        norm = tf.abs(tf.trace(state_dm))
        penalty = tf.pow(norm - 1, 2) # Penalise unnormalised states
        loss = -self.fid - self.hyperparams['gamma'] * penalty

        return loss

    def train(self, epochs):
        """Train the neural network"""
        # Input/output variables not needed for state engineering
        zero = np.array([0])
        return super(StateEngineer, self).train(epochs, zero, zero)
