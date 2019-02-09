import os
import numpy as np
import tensorflow as tf
from qnn.curve_fitter import CurveFitter

n_epochs = 50

f = np.load('curve_fit/training/sinc.npz')
inputs = f['x']
expected_outputs = f['noisy']

sess = tf.Session()
net = CurveFitter(sess, batch_size=50, hyperparams={
    'lr_decay_steps': (inputs.size // 50) * 5
})
sess.run(tf.global_variables_initializer())

losses, lrs = net.train(n_epochs, inputs, expected_outputs)

true_outputs = f['true']
predictions = net.make_prediction(inputs)

save_dir = net.save("save", prefix="CF ")
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
