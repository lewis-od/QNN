import abc
import strawberryfields as sf
import tensorflow as tf
from strawberryfields.ops import *

class QNNBase(metaclass=abc.ABCMeta):
    def __init__(self, sess, trunc=25, n_layers=2, n_modes=2, learning_rate=0.05):
        self.sess = sess
        self.trunc = trunc
        self.n_layers = n_layers
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10, 0.9)
        self.eng, self.q = sf.Engine(n_modes)
        if getattr(self, 'hyperparams', None) is None: self.hyperparams = {}

        # Input values
        self.x = tf.placeholder(tf.float64, shape=[None])
        # Expected output values
        self.y_ = tf.placeholder(tf.float64, shape=[None])

        self.__init_params()
        self.__build_circuit()

        optimiser = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.min_op = optimiser.minimize(self.loss_fn(), global_step=self.global_step)

    def run_circuit(self, input):
        state = self.eng.run('tf', trunc=self.trunc, eval=False)
        state = self.sess.run(state, feed_dict={ self.input: input })
        return state

    def train(self, epochs, inputs, outputs):
        loss = self.loss_fn()
        for step in range(epochs):
            feed_dict = self.hyperparams
            feed_dict[self.x] = inputs
            feed_dict[self.y_] = outputs
            loss_val, _ = self.sess.run([loss, self.min_op], feed_dict=feed_dict)
            print("{} : Loss = {}".format(step, loss_val))
            if step % 10 == 0:
                lr_val = self.sess.run([self.learning_rate])
                print("Learning rate: {}".format(lr_val))

    def get_parameters(self):
        bs, r_vals, a_vals = self.sess.run([self.b_splitters, self.rs, self.alphas])
        return { 'beam_splitters': bs, 'squeezing': r_vals, 'displacement': a_vals }

    @abc.abstractmethod
    def loss_fn(self):
        """Return loss value calculated using self.x and self.y_"""
        pass

    def __init_params(self):
        self.b_splitters = tf.Variable(initial_value=tf.random_uniform([self.n_layers, 2], maxval=2*np.pi),
            dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi))
        self.rs = tf.Variable(initial_value=tf.random_uniform([self.n_layers, 2], minval=-1.4, maxval=1.4),
            dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4))
        self.alphas = tf.Variable(initial_value=tf.random_normal([self.n_layers, 2], mean=1, stddev=4),
            dtype=tf.float32)

    @abc.abstractmethod
    def build_encoder(self):
        """Builds part of circuit that encodes classical info as a quantum state"""
        pass

    def __build_circuit(self):
        with self.eng:
            # Encode classical info as quantum state
            self.build_encoder()
            # Run it through the network
            for n in range(self.n_layers):
                self.__create_layer(n)

    def __create_layer(self, n):
        # Ancilla state
        Fock(1) | self.q[0]

        # Interferometer
        BSgate(self.b_splitters[n][0], -np.pi/4) | (self.q[0], self.q[1])

        # Squeezing
        Sgate(self.rs[n][0]) | self.q[0]
        Sgate(self.rs[n][1]) | self.q[1]

        # Interferometer
        BSgate(self.b_splitters[n][1], -np.pi/4) | (self.q[0], self.q[1])

        # Displacement
        Dgate(self.alphas[n][0]) | self.q[0]
        Dgate(self.alphas[n][1]) | self.q[1]

        # Measure ancilla mode
        MeasureFock(select=0) | self.q[0]
