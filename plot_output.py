import os
import ast
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from datetime import datetime
from strawberryfields.ops import *

# ----- Parse command-line args -----

if len(os.sys.argv) < 2:
    print("Please provide a folder containing results.")
    os.sys.exit(1)

res_folder = os.sys.argv[1]


# ----- Load in hyperparameters -----

hyper_loc = os.path.join(res_folder, 'hyperparams.txt')
with open(hyper_loc, 'r') as hyper_file:
    hyper_str = hyper_file.readline() # Hyperparam dict on first line
    # Name of training file on second line
    train_file_name = hyper_file.readline().split(': ')[-1]

train_file_name = train_file_name.strip()
train_file = os.path.join(res_folder, os.path.pardir, os.path.pardir,
    'training', train_file_name)

# Load in training data
with np.load(train_file) as f:
    training_in = f['x']
    training_out = f['y']
    true_in = f['x_true']
    true_out = f['y_true']

# Load in network outputs
output_file = os.path.join(res_folder, 'output.npz')
with np.load(output_file) as f:
    predictions_in = f['inputs']
    predictions_out = f['predictions']


# ----- Plot results -----

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot predictions, training data, and "true" (non-noisy) curve
true_curve, = plt.plot(true_in, true_out, c='g')
predicted_scatter = plt.scatter(predictions_in, predictions_out,
    c='r', marker='x')
input_scatter = plt.scatter(training_in, training_out, c='b', marker='o')

plt.legend([true_curve, input_scatter, predicted_scatter],
    ['True Curve', 'Training Data', 'Prediction'], scatterpoints=1)
plt.xlabel("$x$", fontsize=18)

y_str = r'$f(x)$'
if 'sinc' in train_file:
    y_str = r'sinc$(x)$'
elif 'sin' in train_file:
    y_str = r'$\sin (x)$'
elif 'x_cubed' in train_file:
    y_str = r'$x^3$'

plt.ylabel(y_str, fontsize=18)

save_path = os.path.join(res_folder, 'plot.eps')
plt.savefig(save_path, bbox_inches='tight')
print("Saved plot to: " + save_path)

plt.show()
