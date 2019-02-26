import os
import numpy as np

n_states = 5000 # Number of states to generate
should_plot = True

# Generate 50% Gaussian and 50% non-Gaussian states
n_gaussian = n_states // 2
n_non_gaussian = n_states - n_gaussian

def rand_complex(r_max, n=1):
    """Returns n random complex numbers with |z| < r_max"""
    r = np.random.rand(n) * r_max
    theta = np.random.rand(n) * np.pi * 2
    z = r * np.exp(1j*theta)
    if n == 1:
        return z[0]
    return z

# Labels for training set 0 => Gaussian, 1 => non-Gaussian
labels = np.zeros(n_states)
labels[n_gaussian:] = np.ones(n_non_gaussian)
# Array to hold state data
states = np.zeros([n_states, 3], dtype=np.complex64)

squeezing = rand_complex(1.4, n_states)
displacement = rand_complex(3.0, n_states)
kappas = np.random.rand(n_non_gaussian)

states[:, 0] = squeezing
states[:, 1] = displacement
states[n_gaussian:, 2] = kappas

# Create save directory if it doesn't exist
save_dir = os.path.join(".", "data")
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Save the generated states and their labels
save_file = os.path.join(save_dir, "states.npz")
np.savez(save_file, states=states, labels=labels)

if should_plot:
    from test_truncation import test_truncations
    test_truncations(states)
