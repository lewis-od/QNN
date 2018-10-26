import time
import numpy as np
import strawberryfields as sf
import matplotlib.pyplot as plt
from qutip import Qobj
from qutip.visualization import plot_wigner
from strawberryfields.ops import *

# Helper functions
def rand_angle():
    return np.random.rand() * np.pi * 2

def rand_complex(max_r):
    r = np.random.ranf() * max_r
    theta = rand_angle()
    return r * np.exp(1j*theta)

# Parameter values for the circuit
thetas = np.array([rand_angle() for _ in range(4)])
b_splitters = np.array([(rand_angle(), rand_angle()) for _ in range(2)])
rs = np.array([rand_complex(1.4) for _ in range(2)])
alphas = np.array([rand_complex(5.0) for _ in range(2)])
alpha_in = 0.5 + 2j;

print("Rotations:")
print("    {}".format(thetas))
print("Beam Splitters:")
print("    {}".format(b_splitters))
print("Squeezing:")
print("    {}".format(rs))
print("Displacements:")
print("    {}".format(alphas))

eng, q = sf.Engine(2)
with eng:
    Fock(1) | q[0]
    Coherent(alpha_in) | q[1]

    # Interferometer
    Rgate(thetas[0]) | q[0]
    Rgate(thetas[1]) | q[1]
    BSgate(*b_splitters[0]) | (q[0], q[1])

    # Squeezing
    Sgate(rs[0]) | q[0]
    Sgate(rs[1]) | q[1]

    # Interferometer
    Rgate(thetas[2]) | q[0]
    Rgate(thetas[3]) | q[1]
    BSgate(*b_splitters[1]) | (q[0], q[1])

    # Displacement
    Dgate(alphas[0]) | q[0]
    Dgate(alphas[1]) | q[0]

    # Measurement
    MeasureFock(select=0) | q[0]

# Run the circuit
start = time.time()
state = eng.run('fock', cutoff_dim=25)
end = time.time()
dt = end - start
print("Time to simulate: {}".format(dt))
# Trace out the ancilla mode
state_dm = state.reduced_dm(1)

# Plot Wigner function of output state
plot_wigner(Qobj(state_dm), colorbar=True)
plt.show()

# TODO: Use qutip.continuous_variables.covariance_matrix to calc cov mat of state
# Also calc displacement vec, and plot wigner of corresponding Gaussian state for comparison
# Calc fidelity between output and closest Gaussian state
