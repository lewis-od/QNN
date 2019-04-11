import os
import numpy as np
import matplotlib.pyplot as plt

n_ancilla = 1
n_post = 0
should_save = True

data_sets = ['sin', 'sinc', 'x_cubed']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(1, 3, figsize=(8*3, 5))
plt.suptitle(r'$(n_{\mathrm{in}}, n_{\mathrm{post}}) = ('
    + str(n_ancilla) + ', ' + str(n_post) + r')$',
    fontsize=32)

for k in range(3):
    res_folder = os.path.join(os.curdir, 'results',
        '{}-{}-{}'.format(data_sets[k], n_ancilla, n_post))

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

    # Plot predictions, training data, and "true" (non-noisy) curve
    true_curve, = axs[k].plot(true_in, true_out, c='g')
    predicted_scatter = axs[k].scatter(predictions_in, predictions_out,
        c='r', marker='x')
    input_scatter = axs[k].scatter(training_in, training_out, c='b', marker='o')

    axs[k].set_xlabel("Input", fontsize=26)
    if k == 0:
        axs[k].set_ylabel("Output", fontsize=26)

if should_save:
    save_path = os.path.join('results', "fock-{}-{}.eps".format(n_ancilla, n_post))
    plt.savefig(save_path, bbox_inches='tight')
    print("Saved plot to: " + save_path)

plt.show()
