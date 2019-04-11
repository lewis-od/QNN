import os
import numpy as np
import matplotlib.pyplot as plt

should_save = True

data_sets = ['sin', 'sinc', 'x_cubed']
n_post = 0

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(8*3, 5))

for k in range(3):
    res_folder = os.path.join(os.curdir, 'results', '{}-{}'.format(data_sets[k], n_post))

    if len(os.sys.argv) > 2:
        should_save = bool(int(os.sys.argv[2]))
        qualifier = "Will" if should_save else "Won't"
        print(qualifier + " save plot")

    train_file = os.path.join(res_folder, os.path.pardir, os.path.pardir,
        'training', data_sets[k] + '.npz')

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

    plt.subplot(1, 3, k+1)

    # Plot predictions, training data, and "true" (non-noisy) curve
    true_curve, = plt.plot(true_in, true_out, c='g')
    predicted_scatter = plt.scatter(predictions_in, predictions_out,
        c='r', marker='x')
    input_scatter = plt.scatter(training_in, training_out, c='b', marker='o')

    plt.xlabel("Input", fontsize=26)
    if k == 0:
        plt.ylabel("Output", fontsize=26)

if should_save:
    save_path = os.path.join('results', "cat-{}.eps".format(n_post))
    plt.savefig(save_path, bbox_inches='tight')
    print("Saved plot to: " + save_path)

plt.show()
