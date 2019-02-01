import abc
import strawberryfields as sf
import tensorflow as tf
from strawberryfields.ops import *

class QNNBase(metaclass=abc.ABCMeta):
    def __init__(self, sess, trunc=25, n_layers=2, n_modes=2, learning_rate=0.05):
        """
        Setup the neural network
        :param sess: A Tensorflow session
        :param trunc: The truncation to use in simulation of the quantum circuit
        :param n_layers: How many layers the neural network has
        :param n_modes: How many modes to use
        :param learning_rate: The initial learning rate (decays exponentially during training)
        """
        self.sess = sess
        self.trunc = trunc
        self.n_layers = n_layers
        # Keep track of how many training steps have been performed (for learning rate decay)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10, 0.9)
        # Initialse strawberry fields
        self.eng, self.q = sf.Engine(n_modes)
        # Create empty hyperparams dict if it hasn't already been set
        if getattr(self, 'hyperparams', None) is None: self.hyperparams = {}

        # Input values
        self.x = tf.placeholder(tf.float64, shape=[None])
        # Expected output values
        self.y_ = tf.placeholder(tf.float32, shape=[None])

        self.__init_params()
        self.__build_circuit()

        # Use stochastic gradiet descent
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # Minimise the loss function
        self.min_op = optimiser.minimize(self.loss_fn(), global_step=self.global_step)

    def run_circuit(self, input):
        """Runs the circuit with a given input value"""
        state = self.eng.run('tf', trunc=self.trunc, eval=False)
        state = self.sess.run(state, feed_dict={ self.x: [input], self.y_: [0] })
        return state

    def train(self, epochs, inputs, outputs):
        """
        Train the neural network
        :param epochs: The number of epochs to train for
        :param inputs: Array of input data
        :param outputs: Array of expected output data
        """
        loss = self.loss_fn()
        for step in range(epochs):
            feed_dict = self.hyperparams
            # Append inputs and outputs to hyperparams dict
            feed_dict[self.x] = inputs
            feed_dict[self.y_] = outputs
            # Print the loss value and run the minimisation operation
            loss_val, _ = self.sess.run([loss, self.min_op], feed_dict=feed_dict)
            print("{} : Loss = {}".format(step, loss_val))
            if step % 10 == 0:
                # Print the learning rate every 10 steps
                lr_val = self.sess.run(self.learning_rate)
                print("Learning rate: {}".format(lr_val))

    def get_parameters(self):
        """Returns the parameters of the neural network"""
        bs, r_vals, a_vals = self.sess.run([self.b_splitters, self.rs, self.alphas])
        return { 'beam_splitters': bs, 'squeezing': r_vals, 'displacement': a_vals }

    @abc.abstractmethod
    def loss_fn(self):
        """Returns loss value calculated using self.eng self.y_"""
        pass

    def __init_params(self):
        """Initialises all network parameters"""
        # Beam splitters - 2 per layer
        self.b_splitters = tf.Variable(initial_value=tf.random_uniform([self.n_layers, 2], maxval=2*np.pi),
            dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi))
        # Squeezing parameters - 2 per layer (1 on each mode)
        self.rs = tf.Variable(initial_value=tf.random_uniform([self.n_layers, 2], minval=-1.4, maxval=1.4),
            dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4))
        # Displacement parameters - 2 per layer (1 on each mode)
        self.alphas = tf.Variable(initial_value=tf.random_normal([self.n_layers, 2], mean=1, stddev=4),
            dtype=tf.float32)

    def __build_circuit(self):
        """Builds the quantum circuit that implements the neural network"""
        with self.eng:
            # Encode classical info as quantum state
            self.build_encoder()
            # Run it through the network
            for n in range(self.n_layers):
                self.__build_layer(n)

    @abc.abstractmethod
    def build_encoder(self):
        """
        Builds part of circuit that encodes classical info as a quantum state.
        This is placed at the beginning of the circuit, and should encode the
        value self.x as a quantum state.
        """
        pass

    def __build_layer(self, n):
        """
        Builds 1 layer of the neural network
        :param n: Index denoting which layer of the neural network is being built
        """
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
