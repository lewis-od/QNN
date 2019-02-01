# Parse the fidelity values and learning rate from the neural network output,
# and plot them
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print("Please specify the file to parse and the number of epochs used.")
    sys.exit(1)

# TODO: Parse this from file
epochs = int(sys.argv[2])

fid_vals = np.zeros(epochs)
i = 0

lr_vals = np.zeros(epochs//5)
j = 0

with open(sys.argv[1], 'r') as f:
    for line in f:
        parts = line.split('=')
        if len(parts) > 1:
            fid_vals[i] = np.float64(parts[-1])
            i += 1
        else:
            parts = line.split('Learning rate: ')
            if len(parts) > 1:
                lr_vals[j] = np.float64(parts[-1])
                j += 1
        if i == epochs:
            break

# plt.subplot(1, 2, 1)
plt.plot(fid_vals)
plt.xticks(np.arange(0, epochs, 1), rotation=45)
plt.xlabel("Epoch")
plt.ylabel("Fidelity")

# plt.subplot(1, 2, 2)
# plt.plot(np.arange(0, epochs, 5), lr_vals)
# plt.xticks(np.arange(0, epochs, 1), rotation=45)
# plt.xlabel("Epoch")
# plt.ylabel("Learning Rate")
# plt.show()
