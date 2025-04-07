import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def pauli_to_numeric(pauli_str):
    """
    Convert a Pauli string (e.g., "IXZY...") into an array of numeric codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    Returns a numpy array of shape (len(pauli_str),).
    """
    mapping = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}
    numeric = [mapping.get(ch, 0) for ch in pauli_str]
    return np.array(numeric)

def plot_spacetime(pauli_strings, cell_count, return_fig=False):
    """
    Plot the spacetime diagram of the QCA simulation.
    
    - pauli_strings: list of strings (each of length cell_count) representing the state at each time step.
    - return_fig: if True, return the figure object instead of showing it (for Streamlit)
    """
    time_steps = len(pauli_strings)
    data = np.zeros((time_steps, cell_count), dtype=int)
    for t, s in enumerate(pauli_strings):
        data[t, :] = pauli_to_numeric(s)
    
    # Define a discrete colormap: I: white, X: red, Z: blue, Y: green.
    cmap = ListedColormap(["white", "red", "blue", "green"])
    
    # Create a square figure with equal cell sizes for time and space.
    cell_size = 0.5  # Size of each cell in inches
    fig_width = cell_size * cell_count
    fig_height = cell_size * time_steps
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    im = ax.imshow(data, cmap=cmap, interpolation="nearest", aspect='equal')
    
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xticks(np.arange(cell_count), labels=range(cell_count))
    ax.set_yticks(np.arange(time_steps), labels=range(time_steps))
    
    ax.set_xlabel("Cell position")
    ax.set_ylabel("Time step")
    ax.set_title("1D Clifford QCA Spacetime Diagram")
    
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['I', 'X', 'Z', 'Y'])
    cbar.set_label("Pauli Operator")
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
