# Runs test to check if input states are normalised
import numpy as np
import strawberryfields as sf
import tensorflow as tf
from strawberryfields.ops import Sgate, Dgate, Vgate

def test_truncations(states, truncations=[10, 15, 20]):
    batch_size = states.shape[0] // 2

    x = tf.placeholder(tf.float32, shape=[batch_size, 3])

    eng, q = sf.Engine(1)
    with eng:
        Sgate(x[:, 0]) | q[0]
        Dgate(x[:, 1]) | q[0]
        Vgate(x[:, 2]) | q[0]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    traces = np.zeros([batch_size, 2, len(truncations)])
    for n in range(len(truncations)):
        state = eng.run('tf', eval=False, batch_size=batch_size, cutoff_dim=truncations[n])
        trace = tf.real(state.trace())

        traces[:, 0, n] = sess.run(trace, feed_dict={
            x: states[:batch_size, :]
        })
        traces[:, 1, n] = sess.run(trace, feed_dict={
            x: states[2500:2500+batch_size, :]
        })
        print("Evaluated T={}".format(truncations[n]))

    import matplotlib.pyplot as plt
    n_plots = 2 * len(truncations)
    for n in range(n_plots):
        plt.subplot(len(truncations), 2, n+1)
        plt.hist(traces[:, n % 2, n // 2], bins=25)
        plt.xlabel("Trace")
        plt.ylabel("Frequency")
        plt.xlim([0.5, 1.0])
        plt.ylim([0, batch_size])
        if n == 0:
            plt.title("Non-Gaussian States")
        elif n == 1:
            plt.title("Gaussian States")
    plt.show()

if __name__ == '__main__':
    f = np.load('data/states.npz')
    state_params = f['states']
    test_truncations(state_params)
