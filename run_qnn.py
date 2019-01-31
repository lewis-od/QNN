import tensorflow as tf
from qnn.states import cubic_phase
from qnn.qnn import QNN

sess = tf.Session()
target_state = cubic_phase(25, 0.005, -1.0).full()
net = QNN(sess, target_state)
sess.run(tf.global_variables_initializer())
net.train(15)

params = net.get_parameters()
print(params)
