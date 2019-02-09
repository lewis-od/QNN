import os
import numpy as np
import tensorflow as tf
from qnn.curve_fitter import CurveFitterX

n_epochs = 200

training_file = 'curve_fit/training/sinc.npz'
if len(os.sys.argv) == 2:
    training_file = os.sys.argv[1]
print("Loading training data from: " +  training_file)

f = np.load(training_file)
inputs = f['x']
expected_outputs = f['noisy']

sess = tf.Session()
net = CurveFitterX(sess, batch_size=50, hyperparams={
    'lr_initial': 0.5,
    'lr_decay_steps': (inputs.size // 50) * 5
})
sess.run(tf.global_variables_initializer())

losses, lrs = net.train(n_epochs, inputs, expected_outputs)

true_outputs = f['true']
predictions = net.make_prediction(inputs)

save_dir = net.save("save", prefix="CFX ")
np.savez(os.path.join(save_dir, 'output.npz'), hyperparams=net.hyperparams,
    loss=losses, learning_rate=lrs)
print("Saved to: " + save_dir)

import matplotlib.pyplot as plt

plt.subplot(1, 3, 1)
plt.plot(inputs, true_outputs)
exp_plt = plt.scatter(inputs, expected_outputs, c='b', marker='o')
pred_plt = plt.scatter(inputs, predictions, c='r', marker='x')
plt.legend([exp_plt, pred_plt],
    ['Training Data', 'Network Predictions'])

plt.subplot(1, 3, 2)
plt.plot(np.arange(n_epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 3)
plt.plot(np.arange(n_epochs), lrs)
plt.xlabel("Epoch")
plt.ylabel("Learning rate")

plt.show()
