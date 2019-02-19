import os
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from states import ONstate, SqueezedCatstate

n_states = 5000
truncation = 10

eng, q = sf.Engine(1)

n_gaussian = n_states // 2
n_non_gaussian = n_states - n_gaussian

def rand_complex(r_max, n=1):
    r = np.random.rand(n) * r_max
    theta = np.random.rand(n) * np.pi * 2
    z = r * np.exp(1j*theta)
    if n == 1:
        return z[0]
    return z

labels = np.zeros(n_states)
labels[n_gaussian:] = np.ones(n_non_gaussian)
states = np.zeros([n_states, truncation], dtype=np.complex64)

squeezing = rand_complex(1.4, n_gaussian)
displacement = rand_complex(3.0, n_gaussian)
for i in range(n_gaussian):
    with eng:
        Dgate(displacement[i]) | q[0]
        Sgate(squeezing[i]) | q[0]
    state = eng.run('fock', cutoff_dim=truncation).ket()
    states[i, :] = state
    eng.reset()

state_types = np.zeros(5)
for i in range(n_non_gaussian):
    type = np.random.randint(0, 5)
    state_types[type] += 1
    Gate = None
    if type == 0: # Cat state
        alpha = rand_complex(2.0)
        p = np.random.randint(0, 2)
        Gate = Catstate(alpha, p)
    elif type == 1: # Cubic phase state
        gamma = np.random.rand() * 0.1
        Gate = Vgate(gamma)
    elif type == 2: # Fock state
        n = np.random.randint(truncation)
        Gate = Fock(n)
    elif type == 3: # Squeezed cat
        alpha = rand_complex(2.0)
        p = np.random.randint(0, 2)
        r = rand_complex(1.4)
        Gate = SqueezedCatstate(alpha, p, r)
    elif type == 4: # ON state
        n = np.random.randint(truncation)
        delta = np.random.rand()
        Gate = ONstate(n, delta)
    with eng:
        Gate | q[0]
    state = eng.run('fock', cutoff_dim=10).ket()
    states[n_gaussian+i, :] = state
    eng.reset()

save_dir = os.path.join(".", "data")
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

state_names = ['cat', 'cp', 'fock', 'sq_cat', 'ON']
state_types = dict(zip(state_names, state_types))

save_file = os.path.join(save_dir, "states.npz")
np.savez(save_file, states=states, labels=labels, types=state_types)
