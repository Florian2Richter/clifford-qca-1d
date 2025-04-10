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
    
    # Define a more appealing colormap with warm brown and complementary colors
    # I: white, X: teal, Z: coral, Y: dark grey
    cmap = ListedColormap(["white", "#008080", "#FF7F50", "#4A4A4A"])

    # Set fixed figure size
    fixed_fig_width = 6  # inches (reduced from 8)
    fixed_fig_height = 4  # inches (reduced from 5)

    # Create figure with white background using fixed size
    fig, ax = plt.subplots(figsize=(fixed_fig_width, fixed_fig_height), facecolor='white')
    ax.set_facecolor('white')
    
    # Create a pcolormesh plot instead of imshow for better control over cell boundaries
    x = np.arange(cell_count + 1)
    y = np.arange(time_steps + 1)
    X, Y = np.meshgrid(x, y)
    
    # Plot the data with pcolormesh
    im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='flat')
    
    # Set the axis limits to match the data and invert y-axis
    ax.set_xlim(0, cell_count)
    ax.set_ylim(time_steps, 0)  # Invert y-axis to make time flow downward
    
    # Calculate tick spacing based on size
    x_tick_spacing = max(1, cell_count // 15)  # Show at most 15 ticks on x-axis
    y_tick_spacing = max(1, time_steps // 15)  # Show at most 15 ticks on y-axis
    
    # Set ticks at cell centers with appropriate spacing
    ax.set_xticks(np.arange(0.5, cell_count, x_tick_spacing))
    ax.set_yticks(np.arange(0.5, time_steps, y_tick_spacing))
    
    # Set tick labels with smaller font size
    ax.set_xticklabels(range(0, cell_count, x_tick_spacing), fontsize=8)
    ax.set_yticklabels(range(0, time_steps, y_tick_spacing), fontsize=8)
    
    # Set axis labels with smaller font size
    ax.set_xlabel("Cell position", fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel("Time step", fontsize=10, fontweight='bold', labelpad=8)
    
    # Set title with smaller font size
    ax.set_title("1D Clifford QCA Spacetime Diagram", fontsize=12, fontweight='bold', pad=15)
    
    # Create a colorbar with smaller font size
    cbar = fig.colorbar(im, ticks=[0.4, 1.2, 2.0, 2.8], orientation='vertical', pad=0.02)
    cbar.ax.set_yticklabels(['I', 'X', 'Z', 'Y'], fontsize=10)
    cbar.set_label("Pauli Operator", fontsize=10, fontweight='bold', labelpad=10)
    
    # Add grid lines at cell boundaries
    ax.grid(True, color='black', linewidth=0.5, linestyle='-', alpha=0.2)
    
    # Adjust layout with more padding
    plt.tight_layout(pad=1.5)
    
    if return_fig:
        return fig
    else:
        plt.show()
