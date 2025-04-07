import numpy as np
import matplotlib.pyplot as plt
from qca.core import build_global_operator, simulate_QCA, pauli_string_to_state
from qca.visualization import plot_spacetime

# Set up simulation parameters
n = 100  # number of cells
T_steps = 50  # number of time steps

# Set up the local rule matrix (example rule)
local_rule = np.array([
    [1, 0, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0]
], dtype=int) % 2

# Create initial state (single X in the middle)
initial_state = np.zeros(2 * n, dtype=int)
center = n // 2
initial_state[2*center] = 1  # Set x-part for an X operator

# Build the global operator and run simulation
global_operator = build_global_operator(n, local_rule)
states, pauli_strings = simulate_QCA(n, T_steps, initial_state, global_operator)

# Create and save the plot
fig = plot_spacetime(pauli_strings, n, return_fig=True)
fig.savefig('docs/images/app_screenshot.png', dpi=300, bbox_inches='tight')
plt.close(fig) 