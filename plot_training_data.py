import os
import numpy as np

if len(os.sys.argv) < 2:
    print("Please provide some training data")
    os.sys.exit(1)

train_file = os.sys.argv[1]
with np.load(train_file) as f:
    training_in = f['x']
    training_out = f['y']
    true_in = f['x_true']
    true_out = f['y_true']

# ----- Plot results -----

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot predictions, training data, and "true" (non-noisy) curve
true_curve, = plt.plot(true_in, true_out, c='g')
input_scatter = plt.scatter(training_in, training_out, c='b', marker='o')

dataset_name = os.path.split(train_file)[-1].split('.')[0]
if dataset_name == 'sin':
    y_label = r'$\sin(x)$'
elif dataset_name == 'sinc':
    y_label = r'sinc$(x)$'
elif dataset_name == 'x_cubed':
    y_label = r'$x^3$'

plt.legend([true_curve, input_scatter], [y_label, 'Training Data'],
    scatterpoints=1, fontsize=16, loc=2)
plt.xlabel("$x$", fontsize=16)

save_path = os.path.join(os.curdir, 'training_{}.eps'.format(dataset_name))
plt.savefig(save_path, bbox_inches='tight')
print("Saved plot to: " + save_path)

plt.show()
