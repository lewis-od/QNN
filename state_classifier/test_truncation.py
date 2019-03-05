# Runs test to check if input states are normalised
import numpy as np
import strawberryfields as sf
import tensorflow as tf
from strawberryfields.ops import Sgate, Dgate, Vgate

def test_truncations(states, truncations=[10, 15, 20]):
    batch_size = states.shape[0]

    x = tf.placeholder(tf.complex64, shape=[batch_size, 3])

    eng, q = sf.Engine(1)
    with eng:
        Sgate(x[:, 0]) | q[0]
        Dgate(x[:, 1]) | q[0]
        Vgate(x[:, 2]) | q[0]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    traces = np.zeros([batch_size, len(truncations)])
    for n in range(len(truncations)):
        state = eng.run('tf', eval=False, batch_size=batch_size, cutoff_dim=truncations[n])
        trace = tf.real(state.trace())

        traces[:, n] = sess.run(trace, feed_dict={
            x: states
        })
        print("Evaluated T={}".format(truncations[n]))

    import matplotlib.pyplot as plt
    for n in range(len(truncations)):
        plt.subplot(len(truncations), 1, n+1)
        plt.hist(traces[:, n], bins=25)
        plt.xlabel("Trace")
        plt.ylabel("Frequency")
    plt.show()

if __name__ == '__main__':
    f = np.load('data/states.npz')
    state_params = f['states']
    test_truncations(state_params)
