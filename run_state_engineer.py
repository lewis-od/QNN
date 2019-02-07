import os
import tensorflow as tf
import numpy as np
from qnn.states import cubic_phase
from qnn.state_engineer import StateEngineer

sess = tf.Session()

target_state = cubic_phase(10, 0.005, -1.0).full()
net = StateEngineer(sess, target_state, hyperparams={ 'lr_initial': 0.1 })

sess.run(tf.global_variables_initializer())
losses, lrs = net.train(100)

fid_val = sess.run(net.fid, feed_dict={ net.x: [0], net.y_: [0] })
print("Final fidelity: {}".format(fid_val))

save_folder = net.save("save", prefix="SE ")
with open(os.path.join(save_folder, 'hyperparams.txt'), 'a') as h_file:
    print('\n', file=h_file)
    print("Fidelity: {}".format(fid_val), file=h_file)
np.savez(os.path.join(save_folder, 'output.npz'),
    losses=losses, learning_rate=lrs)
print("Saved to " + save_folder)
