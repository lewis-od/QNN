import tensorflow as tf
from qnn.states import cubic_phase
from qnn.state_engineer import StateEngineer

sess = tf.Session()

target_state = cubic_phase(25, 0.005, -1.0).full()
net = StateEngineer(sess, target_state)

sess.run(tf.global_variables_initializer())
net.train(10)

params = net.get_parameters()
print(params)

fid_val = sess.run(net.fid, feed_dict={ net.x: [0], net.y_: [0] })
print("Final fidelity: {}".format(fid_val))
