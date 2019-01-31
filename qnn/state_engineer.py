import tensorflow as tf
import strawberryfields as sf
from qnn.base import QNNBase

class StateEngineer(QNNBase):
    def __init__(self, sess, target_state, trunc=25, n_layers=2, gamma=0.5, learning_rate=0.05):
        # Gamma is multiplier for trace penalty
        self.gamma = tf.placeholder(dtype=tf.float32, shape=[], name="gamma")
        self.hyperparams = { self.gamma: gamma }

        # The state to search for
        self.target_state = target_state

        super(StateEngineer, self).__init__(sess, trunc=trunc, n_layers=n_layers, n_modes=2, learning_rate=learning_rate)

    # State engineering has a fixed input state to requires no encoding
    def build_encoder(self):
        return

    def loss_fn(self):
        # Calculate state by simulating circuit
        state = self.eng.run('tf', cutoff_dim=self.trunc, eval=False)
        state_dm = state.reduced_dm(1) # Trace out ancilla mode
        # Convert target_state from vector -> density matrix
        target_dm = self.target_state @ self.target_state.conj().T

        # Calculate the fidelity of the output state with the cubic phase state
        self.fid = tf.abs(tf.trace(state_dm @ target_dm)) # Output of tf.trace should be real, but usually has very small imaginary component
        norm = tf.abs(tf.trace(state_dm))
        penalty = tf.pow(norm - 1, 2) # Penalise unnormalised states
        loss = -self.fid - self.gamma*penalty

        return loss

    def train(self, epochs):
        # Input/output variables not needed for state engineering
        return super(StateEngineer, self).train(epochs, [0], [0])
