import numpy as np
import matplotlib.pyplot as plt
import os
from qca.core import build_global_operator, simulate_QCA, pauli_string_to_state
from matplotlib.colors import ListedColormap
from PIL import Image

# Set up simulation parameters
n = 250  # number of cells
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

# Create a clean thumbnail
plt.figure(figsize=(3, 3), dpi=100)

# Convert Pauli strings to numeric representation for visualization
numeric_data = np.zeros((T_steps, n), dtype=int)
for t in range(T_steps):
    for i in range(n):
        pauli = pauli_strings[t][i]
        if pauli == 'I':
            numeric_data[t, i] = 0
        elif pauli == 'X':
            numeric_data[t, i] = 1
        elif pauli == 'Z':
            numeric_data[t, i] = 2
        elif pauli == 'Y':
            numeric_data[t, i] = 3

# Create a custom colormap
colors = ['white', '#FF5555', '#5555FF', '#55FF55']  # White, Red, Blue, Green
cmap = ListedColormap(colors)

# Plot without axes, labels, or borders
plt.imshow(numeric_data, cmap=cmap, interpolation='nearest', aspect='auto')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

# Create the directory if it doesn't exist
os.makedirs('../florian2richter.github.io/assets/images', exist_ok=True)

# Save the figure with a transparent background
thumbnail_path = '../florian2richter.github.io/assets/images/qca_thumbnail.png'
plt.savefig(thumbnail_path, bbox_inches='tight', pad_inches=0, transparent=True)
plt.close()

# Now resize to exactly 250x250 using PIL
img = Image.open(thumbnail_path)
img = img.resize((250, 250), Image.LANCZOS)
img.save(thumbnail_path)

print(f"Thumbnail saved to {thumbnail_path}") 