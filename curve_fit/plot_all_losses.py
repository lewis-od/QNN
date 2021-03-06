import os
import numpy as np
import matplotlib.pyplot as plt

n_ancilla = 1
n_post = 0
should_save = True

data_sets = ['sin', 'sinc', 'x_cubed']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.figure(figsize=(8*3, 5))

for k in range(3):
    res_folder = os.path.join(os.curdir, 'results',
        '{}-{}-{}'.format(data_sets[k], n_ancilla, n_post))

    if len(os.sys.argv) > 2:
        should_save = bool(int(os.sys.argv[2]))
        qualifier = "Will" if should_save else "Won't"
        print(qualifier + " save plot")

    # Load in network outputs
    output_file = os.path.join(res_folder, 'output.npz')
    with np.load(output_file) as f:
        loss = f['loss']

    # plt.subplot(1, 3, k+1)

    # Plot losses
    plt.plot(loss)

    plt.xlabel("Step", fontsize=16)
    if k == 0:
        plt.ylabel("Loss", fontsize=16)

plt.legend([r'$\sin(x)$', r'$\mathrm{sinc}(x)$', r'$x^3$'], fontsize=16)
plt.title(r'$(n_{\mathrm{in}}, n_{\mathrm{post}}) = ('
    + str(n_ancilla) + ', ' + str(n_post) + r')$',
    fontsize=18)
plt.xlim([0, 2000])

if should_save:
    save_path = os.path.join('results', "losses-{}-{}.eps".format(n_ancilla, n_post))
    plt.savefig(save_path, bbox_inches='tight')
    print("Saved plot to: " + save_path)

plt.show()
