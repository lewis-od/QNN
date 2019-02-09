import abc
import os
import strawberryfields as sf
import tensorflow as tf
from datetime import datetime
from strawberryfields.ops import *

class QNNBase(metaclass=abc.ABCMeta):
    DEFAULT_HYPER_BASE = {
        'lr_initial': 0.2,
        'lr_decay_steps': 5,
        'lr_decay_rate': 0.90,
        'lr_staircase': True,
        'cutoff': 10,
    }
    def __init__(self, sess, batch_size, n_modes=2, n_layers=6, hyperparams={}):
        """
        Setup the neural network
        :param sess: A Tensorflow session
        :param batch_size: The batch size to use in training
        :param n_modes: Number of modes to use in the neural network
            NOTE: If n_modes != 2, then self.__build_layer will need to
            be overridden
        :param n_layers: Number of layers to use
        :param hyperparams: A dict of hyperparams
        """
        self.sess = sess
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hyperparams = self.DEFAULT_HYPER_BASE
        for key, val in hyperparams.items():
            self.hyperparams[key] = val

        # Keep track of how many training steps have been performed (for learning rate decay)
        self.global_step = tf.Variable(0, trainable=False)
        # Exponentially decay learning rate
        self.learning_rate = tf.train.exponential_decay(
            self.hyperparams['lr_initial'],
            self.global_step,
            self.hyperparams['lr_decay_steps']*self.batch_size, # Each batch is 1 step
            self.hyperparams['lr_decay_rate'],
            staircase=self.hyperparams['lr_staircase']
        )
        # Initialise strawberry fields
        self.eng, self.q = sf.Engine(n_modes)

        # Input values
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size])
        # Expected output values
        self.y_ = tf.placeholder(tf.float32, shape=[self.batch_size])

        self.__init_params()
        self.__build_circuit()

        # Use Adam optimiser
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # Minimise the loss function
        self.min_op = optimiser.minimize(self.loss_fn(), global_step=self.global_step)

    def train(self, epochs, inputs, outputs):
        """
        Train the neural network
        :param epochs: The number of epochs to train for
        :param inputs: Array of input data
        :param outputs: Array of expected output data

        :returns losses: List of losses for each epoch
        :returns training_rates: List of training rates for each epoch
        """
        losses = np.zeros(epochs)
        learning_rates = np.zeros(epochs)

        batched_data = self.__batch_generator([inputs, outputs])
        n_batches = inputs.size // self.batch_size
        loss = self.loss_fn()

        for step in range(epochs):
            epoch_loss = 0.0
            for b in range(n_batches):
                batch_in, batch_out = next(batched_data)
                loss_val, _ = self.sess.run([loss, self.min_op], feed_dict={
                    self.x: batch_in,
                    self.y_: batch_out
                })
                epoch_loss += loss_val
            epoch_loss /= n_batches
            print("{}: loss = {}".format(step, epoch_loss))

            lr_val = self.sess.run(self.learning_rate)
            losses[step] = epoch_loss
            learning_rates[step] = lr_val
        return losses, learning_rates

    def get_parameters(self):
        """Returns the parameters of the neural network"""
        bs, r_vals, a_vals = self.sess.run([self.b_splitters, self.rs, self.alphas])
        return { 'beam_splitters': bs, 'squeezing': r_vals, 'displacement': a_vals }

    def save(self, dir, prefix=""):
        """
        Save the tensorflow model and print params & hyperparams to text file.
        Model will be saved in a directory named as the current date and time,
        with the optional prefix prepended.

        :param dir: Base directory to save the model to
        :param prefix: Optional prefix for save folder name

        :returns: The name of the folder the model was saved to
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f_name = prefix + now_str
        save_folder = os.path.join(dir, f_name)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(save_folder, "model.ckpt"))

        n_epochs = self.sess.run(self.global_step)

        with open(os.path.join(save_folder, 'hyperparams.txt'), 'w') as h_file:
            print(self.hyperparams, file=h_file)
            print('\n', file=h_file)
            print("Total epochs: {}".format(n_epochs // self.batch_size),
                file=h_file)
            print('\n', file=h_file)
            print(self.get_parameters(), file=h_file)

        return save_folder


    @abc.abstractmethod
    def loss_fn(self):
        """Returns loss value calculated using self.eng self.y_"""
        return NotImplementedError("QNNBase.loss_fn should be overridden by subclass")

    def __init_params(self):
        """Initialises all network parameters"""
        # Beam splitters - 2 per layer
        self.b_splitters = tf.Variable(initial_value=tf.random_uniform([self.n_layers, 2], maxval=2*np.pi),
            dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 2*np.pi))
        # Squeezing parameters - 2 per layer (1 on each mode)
        self.rs = tf.Variable(initial_value=tf.random_uniform([self.n_layers, 2], minval=-1.4, maxval=1.4),
            dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, -1.4, 1.4))
        # Displacement parameters - 2 per layer (1 on each mode)
        self.alphas = tf.Variable(initial_value=tf.random_normal([self.n_layers, 2], mean=0, stddev=4),
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
        raise NotImplementedError("QNNBase.build_encoder should be overridden by subclass.")

    def __build_layer(self, n):
        """
        Builds 1 layer of the neural network
        :param n: Index denoting which layer of the neural network is being built
        """
        # Ancilla state
        Fock(1) | self.q[1]

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
        MeasureFock(select=0) | self.q[1]

    def __batch_generator(self, arrays):
        """Groups data in the arrays list into batches of size self.batch_size"""
        starts = [0] * len(arrays)
        while True:
            batches = []
            for i, array in enumerate(arrays):
                start = starts[i]
                stop = start + self.batch_size
                diff = stop - array.shape[0]
                if diff <= 0:
                    batch = array[start:stop]
                    starts[i] += self.batch_size
                else:
                    batch = np.concatenate((array[start:], array[:diff]))
                    starts[i] = diff
                batches.append(batch)
            yield batches
