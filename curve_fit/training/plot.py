# Plots the data in a training file
import os
import numpy as np
import matplotlib.pyplot as plt

if len(os.sys.argv) < 2:
    print("Please supply a file containing training data")
    os.sys.exit(1)

f_name = os.sys.argv[1]
f = np.load(f_name)

x = f['x']
y = f['y']
x_true = f['x_true']
y_true = f['y_true']

plt.plot(x_true, y_true)
plt.scatter(x, y, color='r', marker='x')
plt.show()
