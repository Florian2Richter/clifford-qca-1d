import numpy as np
import matplotlib.pyplot as plt
import os
from qca.core import build_global_operator, simulate_QCA, pauli_string_to_state
from qca.visualization import plot_spacetime

# Set up simulation parameters
n = 500  # number of cells
T_steps = 250  # number of time steps

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

# Use absolute path
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'florian2richter.github.io', 'assets', 'images')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'qca_plot.png')
print(f"Saving to {output_path}")
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig) 